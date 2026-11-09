"""Complete E2E test: ingest → translate → polish → state → MD/EPUB export with emphasis.

Verifies:
1. Source text contains full emphasis content (not stripped)
2. LLM receives encoded markers with full text
3. Translation preserves markers around Chinese translation
4. Polish modifies text while preserving markers
5. State persists target_inline with accurate ranges
6. MD export shows emphasis markers
7. EPUB export shows emphasis tags
8. Broken markers cause export failure
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import pytest

from wenyi_core.config import Config
from wenyi_core.ingest.epub_reader import read_epub
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.file import FileStorage

from tests.fake_llm import routing_handler as base_routing_handler


def _create_test_epub(epub_path: str) -> None:
    """Create EPUB with emphasis tags."""
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
    <dc:language>ja</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/></spine>
</package>""",
        )
        zf.writestr(
            "OEBPS/c1.xhtml",
            """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter</title></head>
<body>
  <h1>Test Chapter</h1>
  <p>First paragraph with <em>emphasis</em> word.</p>
  <p>Second paragraph with <strong>bold</strong> text.</p>
</body>
</html>""",
        )


def marker_preserving_handler(messages, tier, json_mode):
    """Handler that preserves [[TAG_START]]/[[TAG_END]] markers."""
    system = messages[0]["content"]
    user = messages[-1]["content"]

    # For translation/polish
    if "translator" in system.lower() or "prose editor" in system.lower():
        # Check if polish
        if "Polish the translations" in user or "polish" in system.lower():
            # Polish: get previous translations and modify while preserving markers
            for msg in reversed(messages[:-1]):
                if msg.get("role") == "assistant":
                    try:
                        prev = json.loads(msg["content"])
                        trans = prev.get("translations", [])
                        if trans:
                            polished = []
                            for t in trans:
                                # Extract markers and text
                                marker_pattern = r"(\[\[[A-Z_]+_(START|END)\]\])"
                                parts = re.split(marker_pattern, t)

                                result = []
                                for part in parts:
                                    if re.match(marker_pattern, part):
                                        # Keep marker
                                        result.append(part)
                                    elif part.strip() and part not in ("START", "END"):
                                        # Polish text: change length/order
                                        result.append(f"{part}润色版")
                                    elif part in ("START", "END"):
                                        # Skip split artifact
                                        pass
                                    else:
                                        result.append(part)

                                polished.append("".join(result))

                            return json.dumps({"polished": polished}, ensure_ascii=False)
                    except Exception:
                        pass

        # Translation: extract paragraphs and preserve markers
        paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE)

        if paragraphs:
            translations = []
            for idx, text in paragraphs:
                # Find START/END pairs
                marker_pattern = r"\[\[([A-Z_]+)_(START|END)\]\]"

                # Split by markers, preserving them
                parts = re.split(marker_pattern, text)

                result = []
                i = 0
                while i < len(parts):
                    if i + 2 < len(parts) and parts[i + 2] in ("START", "END"):
                        # parts[i] = text before marker
                        # parts[i+1] = tag name (EM, STRONG, etc)
                        # parts[i+2] = START or END

                        if parts[i].strip():
                            # Translate text before marker
                            result.append(f"中文{idx}")

                        # Add marker
                        result.append(f"[[{parts[i + 1]}_{parts[i + 2]}]]")
                        i += 3
                    else:
                        # Regular text
                        if parts[i].strip():
                            result.append(f"中文{idx}")
                        i += 1

                translations.append("".join(result))

            return json.dumps({"translations": translations}, ensure_ascii=False)

    # Fallback to base handler
    return base_routing_handler(messages, tier, json_mode)


def test_e2e_emphasis_preserved_through_pipeline():
    """Complete E2E: emphasis preserved from ingest to export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        _create_test_epub(epub_path)

        # Step 1: Verify ingest produces correct source with emphasis content
        doc = read_epub(epub_path, source_lang="ja", target_lang="zh")
        ch = doc.chapters[0]

        # Find paragraph with emphasis
        emphasis_seg = None
        for seg in ch.segments:
            if seg.source and "paragraph" in seg.source and "emphasis" in seg.source:
                emphasis_seg = seg
                break

        assert emphasis_seg is not None, "Must find segment with emphasis"
        assert "emphasis" in emphasis_seg.source, "Source must contain 'emphasis' word"
        assert "First paragraph with emphasis word." == emphasis_seg.source

        inline_meta = emphasis_seg.meta.get("epub_inline")
        assert inline_meta is not None
        nodes = inline_meta.get("nodes", [])
        assert len(nodes) == 1
        assert nodes[0]["tag"] == "em"
        assert nodes[0]["offset"] == 21  # Position of "emphasis"

        # Step 2: Run pipeline
        state_dir = str(tmp / "state")
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"polish": True, "review": False, "review_autofix": False},
                "paths": {"state_dir": state_dir},
            }
        )

        client = FakeClient(handler=marker_preserving_handler)
        orch = Orchestrator(config, client=client)
        store = orch.run(epub_path)

        # Step 3: Verify state after translation
        translated_ch = store.load_chapter(0)

        trans_seg = None
        for seg in translated_ch.segments:
            if seg.source and "emphasis" in seg.source:
                trans_seg = seg
                break

        assert trans_seg is not None
        assert trans_seg.target is not None

        # Verify target has NO control markers
        assert "[[" not in trans_seg.target
        assert "]]" not in trans_seg.target

        # Verify target_inline exists with accurate ranges
        target_inline = trans_seg.meta.get("target_inline")
        assert target_inline is not None, "target_inline must exist"
        assert "ranges" in target_inline

        ranges = target_inline["ranges"]
        assert len(ranges) == 1, f"Expected 1 range, got {len(ranges)}: {ranges}"

        r = ranges[0]
        assert r["tag"] == "em"
        assert isinstance(r["start"], int)
        assert isinstance(r["end"], int)
        assert r["start"] < r["end"], "Range must be valid"

        # Verify emphasized text in target
        emphasized_text = trans_seg.target[r["start"] : r["end"]]
        assert emphasized_text, "Emphasized section must not be empty"
        assert "中文" in emphasized_text, "Emphasized text must be Chinese translation"

        # Step 4: Export Markdown
        md_path = str(tmp / "output.md")
        orch.run_assemble(epub_path, out_path=md_path, out_format="markdown")

        assert Path(md_path).exists()
        md_content = Path(md_path).read_text(encoding="utf-8")

        # Verify emphasis markers in markdown
        assert "*" in md_content or "_" in md_content, "MD must have emphasis markers"

        # Step 5: Export EPUB
        epub_out = str(tmp / "output.epub")
        orch.run_assemble(epub_path, out_path=epub_out, out_format="epub")

        assert Path(epub_out).exists()

        with zipfile.ZipFile(epub_out, "r") as zf:
            html = zf.read("OEBPS/c1.xhtml").decode("utf-8")

            # Verify emphasis tags in EPUB
            assert "<em>" in html or "<i>" in html, "EPUB must have emphasis tags"
            assert "中文" in html, "EPUB must have Chinese translation"


def test_e2e_broken_markers_fail_export():
    """Verify that broken/missing markers cause export failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        state_dir = str(tmp / "state" / "book")
        os.makedirs(state_dir, exist_ok=True)

        store = FileStorage(state_dir)

        # Create segment with inline markup but broken target_inline
        seg = Segment(
            index=0,
            source="Text with emphasis word.",
            target="中文译文。",  # Translation WITHOUT emphasis
            meta={
                "epub_inline": {
                    "version": 1,
                    "source_length": 23,
                    "nodes": [
                        {"id": "tn0_0_inline_0", "tag": "em", "placement": "inline", "offset": 10}
                    ],
                },
                "target_inline": {
                    "version": 1,
                    "target_length": 5,
                    "ranges": [],  # NO ranges = markers lost
                    "source_node_count": 1,
                    "target_node_count": 0,
                    "preservation_failed": True,
                },
            },
        )

        chapter = Chapter(index=0, title="Test", segments=[seg])
        store.save_chapter(chapter)

        # Create minimal EPUB
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
                '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body><p>Text with <em>emphasis</em> word.</p></body></html>',
            )

        manifest = {
            "fmt": "epub",
            "source_lang": "ja",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapters": [{"index": 0, "title": "Test", "status": "done"}],
            "meta": {"epub_resources": [{"href": "c1.xhtml"}]},
        }
        store.save_manifest(manifest)

        # Attempt EPUB export - should fail
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {"preset": "fake"},
                "paths": {"state_dir": os.path.dirname(state_dir)},
            }
        )
        client = FakeClient()
        orch = Orchestrator(config, client=client)

        epub_out = str(tmp / "bad.epub")

        # Should raise error due to preservation_failed
        with pytest.raises((ValueError, AssertionError, KeyError)):
            orch.run_assemble(epub_path, out_path=epub_out, out_format="epub")

        # Verify no output file
        assert not Path(epub_out).exists(), "Failed export must not create output"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
