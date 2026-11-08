"""Simplified E2E translation test with fake LLM and dual-format export verification.

Tests: ingest → translate → polish → persist → reload → export MD+EPUB → verify emphasis/images/state.
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
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.file import FileStorage

from tests.fake_llm import routing_handler


def test_e2e_simple_translation_state_and_export():
    """Simplified E2E: translate simple EPUB → verify state/target_inline → export both formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        state_dir = str(tmp / "state")

        # Create minimal EPUB with emphasis and images
        with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr(
                "META-INF/container.xml",
                """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>""",
            )
            zf.writestr(
                "OEBPS/content.opf",
                """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>
    <item id="img1" href="img.png" media-type="image/png"/>
  </manifest>
  <spine><itemref idref="c1"/></spine>
</package>""",
            )
            zf.writestr(
                "OEBPS/c1.xhtml",
                """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test Chapter</title></head>
<body>
  <h1>Test Chapter</h1>
  <div><img src="img.png" alt="Test Image"/></div>
  <p>First paragraph with <em>emphasis</em> word.</p>
  <p>Second paragraph <strong>bold</strong> text.</p>
</body>
</html>""",
            )
            png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
            zf.writestr("OEBPS/img.png", png_data)

        # Use standard routing_handler with language override
        def handler_wrapper(messages, tier, json_mode):
            system = messages[0]["content"]
            # Override language detection to return English
            if "language detector" in system:
                return json.dumps({"language": "en"}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        # Configure pipeline (use defaults, polish enabled)
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},  # routing_handler expects ja
                "llm": {"preset": "fake"},
                "pipeline": {
                    "polish": True,
                    "review": False,
                    "review_autofix": False,
                },
                "paths": {"state_dir": state_dir},
            }
        )

        client = FakeClient(handler=handler_wrapper)
        orch = Orchestrator(config, client=client)

        # Run translation
        store = orch.run(epub_path)

        # Verify state
        manifest = store.load_manifest()
        assert manifest["fmt"] == "epub"
        assert len(manifest["chapters"]) == 1

        # Reload chapter
        chapter = store.load_chapter(0)
        # routing_handler returns "译X" format
        assert "译" in chapter.segments[0].target or "润" in chapter.segments[0].target

        # Verify target_inline exists
        seg0 = chapter.segments[0]
        target_inline = seg0.meta.get("target_inline")
        assert target_inline is not None, "target_inline must exist"
        assert isinstance(target_inline, dict)

        # Export Markdown
        md_path = str(tmp / "output.md")
        orch.run_assemble(store, out_path=md_path, out_format="markdown")
        assert Path(md_path).exists()

        md_content = Path(md_path).read_text(encoding="utf-8")
        assert "译" in md_content or "润" in md_content
        assert "*" in md_content or "_" in md_content  # Has emphasis
        assert "assets/" in md_content  # Has image reference

        # Check image extracted
        assets_dir = tmp / "assets"
        assert assets_dir.exists()
        imgs = list(assets_dir.glob("*.png"))
        assert len(imgs) >= 1

        # Export EPUB
        epub_out = str(tmp / "output.epub")
        orch.run_assemble(store, out_path=epub_out, out_format="epub")
        assert Path(epub_out).exists()

        # Verify EPUB
        with zipfile.ZipFile(epub_out, "r") as zf:
            names = zf.namelist()
            assert "OEBPS/img.png" in names
            ch_html = zf.read("OEBPS/c1.xhtml").decode("utf-8")
            assert "译" in ch_html or "润" in ch_html
            assert "<em>" in ch_html or "<i>" in ch_html

        # Verify no additional LLM calls during export
        call_count = len(client.calls)
        md_path2 = str(tmp / "output2.md")
        orch.run_assemble(store, out_path=md_path2, out_format="markdown")
        assert len(client.calls) == call_count, "Export should not call LLM"


def test_missing_target_inline_fails_assembly():
    """Verify old state without target_inline fails during EPUB assembly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        state_dir = str(tmp / "state" / "book")
        os.makedirs(state_dir, exist_ok=True)

        store = FileStorage(state_dir)

        # Create segment with inline markup but no target_inline
        seg = Segment(
            index=0,
            source="Test <em>emphasis</em>.",
            target="测试强调。",
            meta={
                "epub_inline": {
                    "version": 1,
                    "source_length": 23,
                    "nodes": [{"id": "p", "tag": "p"}, {"id": "em1", "tag": "em", "placement": "inline"}],
                }
                # Missing target_inline!
            },
        )
        chapter = Chapter(index=0, title="Test", segments=[seg])
        store.save_chapter(chapter)

        # Create minimal EPUB as source
        epub_path = str(tmp / "src.epub")
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
            )
            zf.writestr(
                "OEBPS/content.opf",
                '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Test</dc:title></metadata><manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/></spine></package>',
            )
            zf.writestr(
                "OEBPS/c1.xhtml",
                '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body><p>Test <em>emphasis</em>.</p></body></html>',
            )

        manifest = {
            "fmt": "epub",
            "source_lang": "en",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapters": [{"index": 0, "title": "Test", "status": "done"}],
            "meta": {"epub_resources": [{"href": "c1.xhtml"}]},
        }
        store.save_manifest(manifest)

        # Attempt assembly
        config = Config.from_dict(
            {"language": {"source": "en", "target": "zh"}, "llm": {"preset": "fake"}, "paths": {"state_dir": os.path.dirname(state_dir)}}
        )
        client = FakeClient()
        orch = Orchestrator(config, client=client)

        epub_out = str(tmp / "bad.epub")
        with pytest.raises((ValueError, KeyError, AssertionError)):
            orch.run_assemble(store, out_path=epub_out, out_format="epub")

        assert not Path(epub_out).exists(), "Failed assembly must not create output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
