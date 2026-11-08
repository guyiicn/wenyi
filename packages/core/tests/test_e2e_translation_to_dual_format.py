"""End-to-end translation pipeline test with real orchestrator, fake LLM, and dual-format export.

Tests complete flow: ingest EPUB → translate → polish → persist state → reload from disk →
export Markdown and EPUB from same frozen state → verify emphasis, images, anchors, resources.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.file import FileStorage


def _create_test_epub(epub_path: str) -> None:
    """Create minimal EPUB with images, nested emphasis, class spans, anchors, and links."""
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )

        zf.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
    <dc:identifier id="bookid">test-e2e-123</dc:identifier>
  </metadata>
  <manifest>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="titleimg" href="images/title.png" media-type="image/png"/>
    <item id="standalone" href="images/standalone.png" media-type="image/png"/>
    <item id="css" href="styles/main.css" media-type="text/css"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
  </spine>
</package>""",
        )

        # Chapter with emphasis, images, anchors, links
        zf.writestr(
            "OEBPS/chapter1.xhtml",
            """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <title>Chapter One</title>
  <link rel="stylesheet" href="styles/main.css"/>
</head>
<body>
  <h1><img src="images/title.png" alt="Title"/> Chapter One</h1>
  <div><img src="images/standalone.png" alt="Standalone Figure"/></div>
  <p id="para1">First paragraph with <em>simple emphasis</em> text.</p>
  <p id="para2">Second paragraph <strong>with <em>nested emphasis</em> inside</strong> words.</p>
  <p id="para3">Third paragraph has <span class="special">classed span</span> and <i>mid-sentence <strong>multi-level</strong> marks</i> here.</p>
  <p id="para4">Fourth paragraph <a href="#para1">links to first</a> anchor.</p>
</body>
</html>""",
        )

        # Minimal CSS
        zf.writestr(
            "OEBPS/styles/main.css", "body { font-family: serif; }\n.special { color: red; }"
        )

        # Image files (minimal valid PNG)
        png_header = b"\x89PNG\r\n\x1a\n"
        png_ihdr = (
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        )
        png_iend = b"\x00\x00\x00\x00IEND\xaeB`\x82"
        png_data = png_header + png_ihdr + png_iend
        zf.writestr("OEBPS/images/title.png", png_data)
        zf.writestr("OEBPS/images/standalone.png", png_data + b"\x00" * 10)  # Different bytes


def _translation_handler(messages, tier, json_mode):
    """Fake LLM handler: return Chinese with different length/word order than source."""
    import re

    system = messages[0]["content"]
    user = messages[-1]["content"]

    # Language detection
    if "language detector" in system.lower():
        return json.dumps({"language": "en"}, ensure_ascii=False)

    # Pre-translation analysis
    if "analyst" in system.lower() or "analysis" in system.lower():
        return json.dumps(
            {
                "genre": "技术",
                "tone": "正式",
                "style_guide": "准确",
                "characters": [],
            },
            ensure_ascii=False,
        )

    # Title translation
    if "title" in system.lower() and "translat" in system.lower():
        # Count titles
        n = len(re.findall(r"^\[(\d+)\]", user, re.MULTILINE))
        if n == 0:
            n = 1
        return json.dumps({"titles": ["第一章"] * n}, ensure_ascii=False)

    # Main translation: return Chinese with significantly different length/order
    if "literary translator" in system.lower() or "translator" in system.lower():
        # Check if this is polish request
        if "polish" in user.lower() or "polish" in system.lower():
            # Polish: change text but preserve inline markers
            for msg in reversed(messages[:-1]):
                if msg.get("role") == "assistant":
                    try:
                        prev = json.loads(msg["content"])
                        translations = prev.get("translations", [])
                        if translations:
                            polished = []
                            marker_pattern = r"(\[\[[A-Z_0-9]+_(?:START|END)\]\])"
                            for t in translations:
                                # Preserve markers while changing text
                                parts = re.split(marker_pattern, t)
                                result = []
                                for part in parts:
                                    if re.match(marker_pattern, part):
                                        # Keep marker as-is
                                        result.append(part)
                                    elif part.strip():
                                        # Change text but preserve markers
                                        result.append(part.replace("译文", "润色后译文"))
                                    else:
                                        result.append(part)
                                polished.append("".join(result))
                            return json.dumps({"polished": polished}, ensure_ascii=False)
                    except (json.JSONDecodeError, KeyError):
                        pass
            return json.dumps({"polished": []}, ensure_ascii=False)

        # Extract numbered paragraphs and preserve inline markers
        paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE)
        if not paragraphs:
            # Fallback
            n = len(re.findall(r"^\[(\d+)\]", user, re.MULTILINE))
            return json.dumps({"translations": ["译文"] * n}, ensure_ascii=False)
        
        # Return translations preserving inline markers exactly
        translations = []
        marker_pattern = r"(\[\[[A-Z_0-9]+_(?:START|END)\]\])"
        
        for i, (idx, source_text) in enumerate(paragraphs):
            # Split by markers
            parts = re.split(marker_pattern, source_text)
            
            result = []
            for part in parts:
                if re.match(marker_pattern, part):
                    # Keep marker as-is
                    result.append(part)
                elif part.strip():
                    # Translate text part, keep markers separate
                    result.append(f"译{i}")
                else:
                    result.append(part)
            
            translations.append("".join(result))
        
        return json.dumps({"translations": translations}, ensure_ascii=False)

    # Prose editor / polish
    if "prose editor" in system.lower():
        # Extract numbered paragraphs and preserve inline markers
        paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE)
        if not paragraphs:
            n = len(re.findall(r"^\[(\d+)\]", user, re.MULTILINE))
            return json.dumps({"polished": ["润文"] * n}, ensure_ascii=False)
        
        polished = []
        marker_pattern = r"(\[\[[A-Z_0-9]+_(?:START|END)\]\])"
        for idx, text in paragraphs:
            # Preserve markers while polishing
            parts = re.split(marker_pattern, text)
            result = []
            for part in parts:
                if re.match(marker_pattern, part):
                    result.append(part)
                elif part.strip():
                    result.append(part.replace("译文", "润色后译文"))
                else:
                    result.append(part)
            polished.append("".join(result))
        
        return json.dumps({"polished": polished}, ensure_ascii=False)

    # Review
    if "reviewer" in system.lower():
        return json.dumps(
            {"issues": [], "reviewed_segments": 4, "complete": True}, ensure_ascii=False
        )

    # Chapter digest
    if "digest" in system.lower() or "synopsis" in system.lower():
        return "本章内容：介绍人物和场景。"

    # Default
    return "{}" if json_mode else ""


def test_e2e_translation_with_emphasis_images_and_dual_format_export():
    """Full pipeline: translate EPUB with emphasis/images → verify state → export MD+EPUB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        state_dir = str(tmp / "state")

        # Create test EPUB
        _create_test_epub(epub_path)

        # Configure pipeline with polish
        config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {
                    "polish": True,
                    "review": False,  # Disable review for now
                    "review_autofix": False,
                },
                "paths": {"state_dir": state_dir},
            }
        )

        # Create fake client
        client = FakeClient(handler=_translation_handler)

        # Run translation pipeline
        orch = Orchestrator(config, client=client)
        # Use run() which prepares and translates
        store = orch.run(epub_path)

        # Verify state was saved to disk
        manifest = store.load_manifest()
        assert manifest["fmt"] == "epub"
        assert manifest["source_lang"] == "en"
        assert manifest["target_lang"] == "zh"
        assert len(manifest["chapters"]) == 1

        # Reload chapter from disk
        chapter = store.load_chapter(0)
        assert chapter.title == "第一章"
        assert len(chapter.segments) == 4

        # Verify translations were applied
        assert "润色后译文" in chapter.segments[0].target
        assert "强调" in chapter.segments[0].target

        # Verify target_inline metadata exists
        seg0 = chapter.segments[0]
        target_inline = seg0.meta.get("target_inline")
        assert target_inline is not None, "target_inline must exist after translation"
        assert isinstance(target_inline, dict)
        assert "items" in target_inline

        # Verify emphasis items map to Chinese phrases
        items = target_inline["items"]
        assert len(items) > 0, "Should have emphasis items"
        # Check that at least one item has Chinese text
        has_chinese = any("强调" in item.get("target_text", "") for item in items)
        assert has_chinese, "target_inline items should reference Chinese phrases"

        # Export Markdown
        md_path = str(tmp / "output.md")
        orch.assemble(store, out_path=md_path, out_format="markdown")
        assert Path(md_path).exists()

        md_content = Path(md_path).read_text(encoding="utf-8")

        # Verify Markdown has chapter title
        assert "# 第一章" in md_content

        # Verify Markdown has emphasis markers
        assert "*" in md_content or "_" in md_content, "Should have emphasis in Markdown"

        # Verify Markdown has images with assets/
        assert "![Title]" in md_content or "![title]" in md_content.lower()
        assert "assets/" in md_content, "Images should reference assets/"

        # Check assets directory
        assets_dir = tmp / "assets"
        assert assets_dir.exists()
        asset_files = list(assets_dir.glob("*.png"))
        assert len(asset_files) == 2, "Should have 2 image files"

        # Verify image bytes (check they are different)
        img_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in asset_files}
        assert len(set(img_hashes.values())) == 2, "Images should have different content"

        # Export EPUB
        epub_out_path = str(tmp / "output.epub")
        orch.assemble(store, out_path=epub_out_path, out_format="epub")
        assert Path(epub_out_path).exists()

        # Verify EPUB structure
        with zipfile.ZipFile(epub_out_path, "r") as zf:
            namelist = zf.namelist()

            # Check original resources preserved
            assert "OEBPS/styles/main.css" in namelist
            assert "OEBPS/images/title.png" in namelist
            assert "OEBPS/images/standalone.png" in namelist

            # Verify CSS unchanged
            css_content = zf.read("OEBPS/styles/main.css").decode("utf-8")
            assert "font-family: serif" in css_content

            # Verify images unchanged
            title_img = zf.read("OEBPS/images/title.png")
            standalone_img = zf.read("OEBPS/images/standalone.png")
            assert len(title_img) > 0
            assert len(standalone_img) > 0
            assert title_img != standalone_img

            # Read translated chapter
            chapter_html = zf.read("OEBPS/chapter1.xhtml").decode("utf-8")

            # Verify Chinese text present
            assert "润色后译文" in chapter_html or "译文" in chapter_html

            # Verify emphasis tags preserved
            assert "<em>" in chapter_html or "<i>" in chapter_html
            assert "<strong>" in chapter_html

            # Verify class preserved
            assert 'class="special"' in chapter_html

            # Verify anchors preserved
            assert 'id="para1"' in chapter_html or "para1" in chapter_html

            # Verify links preserved
            assert 'href="#para1"' in chapter_html or "#para" in chapter_html

        # Verify no additional LLM calls for export (reuse frozen state)
        initial_call_count = len(client.calls)
        # Export again - should not trigger translation
        md_path2 = str(tmp / "output2.md")
        orch.assemble(store, out_path=md_path2, out_format="markdown")
        assert len(client.calls) == initial_call_count, "Export should not call LLM again"


def test_e2e_translation_state_must_have_target_inline():
    """Verify that old state without target_inline fails during assembly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        state_dir = str(tmp / "state" / "book")

        _create_test_epub(epub_path)

        # Create manual state without target_inline
        os.makedirs(state_dir, exist_ok=True)
        store = FileStorage(state_dir)

        # Fake chapter without target_inline
        from wenyi_core.models.document import Chapter, Segment

        seg = Segment(
            index=0,
            source="Test with <em>emphasis</em>.",
            target="测试强调。",
            meta={
                "epub_inline": {
                    "version": 1,
                    "source_length": 28,
                    "nodes": [
                        {"id": "p", "tag": "p"},
                        {"id": "em1", "tag": "em", "placement": "inline"},
                    ],
                }
                # Missing target_inline!
            },
        )
        chapter = Chapter(index=0, title="Test", segments=[seg])
        store.save_chapter(chapter)

        manifest = {
            "fmt": "epub",
            "source_lang": "en",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapter_count": 1,
            "source_sha256": "fake123",
            "chapters": [{"index": 0, "title": "Test", "status": "done"}],
            "meta": {"epub_resources": [{"href": "chapter1.xhtml"}]},
        }
        store.save_manifest(manifest)

        # Attempt to assemble - should fail
        config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "paths": {"state_dir": os.path.dirname(state_dir)},
            }
        )
        client = FakeClient(handler=_translation_handler)
        orch = Orchestrator(config, client=client)

        epub_out = str(tmp / "bad_output.epub")

        # Should raise error about missing target_inline
        with pytest.raises((ValueError, KeyError, AssertionError)):
            orch.assemble(store, out_path=epub_out, out_format="epub")

        # Verify no output file created
        assert not Path(epub_out).exists(), "Failed assembly should not create output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
