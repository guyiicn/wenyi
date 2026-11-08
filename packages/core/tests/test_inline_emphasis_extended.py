"""Extended inline emphasis tests: failure paths, nesting, CSS styles, and autofix interaction.

Tests real pipeline execution with FakeClient returning broken responses:
- Missing END markers
- Mismatched nesting
- Polish removing markers
- Legacy state without target_inline
- Nested em/strong combinations
- CSS font-style:italic/font-weight:bold spans
- Autofix modifying translated text with inline metadata
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
from wenyi_core.assemble.inline_validation import InlinePreservationError
from wenyi_core.llm.providers.fake import FakeClient
from tests.fake_llm import routing_handler as base_routing_handler
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.file import FileStorage


def test_broken_markers_missing_end():
    """Missing END marker from translation causes decode failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create EPUB with emphasis
        epub_path = str(tmp / "test.epub")
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
            )
            zf.writestr(
                "OEBPS/content.opf",
                """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
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
  <p>Text with <em>emphasis</em> word.</p>
</body>
</html>""",
            )

        state_dir = str(tmp / "state")
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"polish": False, "review": False, "review_autofix": False},
                "paths": {"state_dir": state_dir},
            }
        )

        # Handler drops END markers
        def broken_handler(messages, tier, json_mode):
            system = messages[0]["content"]
            user = messages[-1]["content"]

            if "translator" in system.lower():
                paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE)
                if paragraphs:
                    translations = []
                    for idx, text in paragraphs:
                        # Drop END markers but keep START
                        result = text.replace("[[EM_END]]", "").replace("[[STRONG_END]]", "")
                        translations.append(result)
                    return json.dumps({"translations": translations}, ensure_ascii=False)

            return base_routing_handler(messages, tier, json_mode)

        client = FakeClient(handler=broken_handler)
        orch = Orchestrator(config, client=client)

        # Translation with broken markers should set preservation_failed
        store = orch.run(epub_path)

        # Verify preservation_failed is set
        ch = store.load_chapter(0)
        emphasis_seg = next((s for s in ch.segments if s.source and "emphasis" in s.source), None)
        assert emphasis_seg is not None
        target_inline = emphasis_seg.meta.get("target_inline", {})
        assert target_inline.get("preservation_failed") == True, (
            "Must mark preservation_failed when markers broken"
        )

        # Export must fail
        md_path = str(tmp / "output.md")
        with pytest.raises(InlinePreservationError):
            orch.run_assemble(epub_path, out_path=md_path, out_format="markdown")

        assert not Path(md_path).exists(), "Failed export must not create output file"


def test_broken_markers_mismatched_nesting():
    """Mismatched nesting (closing inner before outer) causes decode failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create EPUB with nested emphasis
        epub_path = str(tmp / "nested.epub")
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebeps-package+xml"/></rootfiles></container>',
            )
            zf.writestr(
                "OEBPS/content.opf",
                """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
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
  <p>Text with <strong>bold and <em>nested emphasis</em> here</strong> end.</p>
</body>
</html>""",
            )

        state_dir = str(tmp / "state")
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"polish": False, "review": False, "review_autofix": False},
                "paths": {"state_dir": state_dir},
            }
        )

        # Handler returns mismatched nesting
        def mismatched_handler(messages, tier, json_mode):
            system = messages[0]["content"]
            user = messages[-1]["content"]

            if "translator" in system.lower():
                paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE)
                if paragraphs:
                    translations = []
                    for idx, text in paragraphs:
                        # Intentionally mismatch: close STRONG before EM
                        if "STRONG" in text and "EM" in text:
                            result = "中文[[STRONG_START]]粗体[[EM_START]]强调[[STRONG_END]]错误[[EM_END]]结束"
                        else:
                            result = f"中文{idx}"
                        translations.append(result)
                    return json.dumps({"translations": translations}, ensure_ascii=False)

            return base_routing_handler(messages, tier, json_mode)

        client = FakeClient(handler=mismatched_handler)
        orch = Orchestrator(config, client=client)

        # Should complete with preservation_failed
        store = orch.run(epub_path)

        ch = store.load_chapter(0)
        nested_seg = next((s for s in ch.segments if s.source and "nested" in s.source), None)
        assert nested_seg is not None
        target_inline = nested_seg.meta.get("target_inline", {})
        assert (
            target_inline.get("preservation_failed") == True
            or len(target_inline.get("validation_warnings", [])) > 0
        )


def test_polish_removes_markers():
    """Polish removing markers causes preservation_failed and export rejection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        epub_path = str(tmp / "test.epub")
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebeps-package+xml"/></rootfiles></container>',
            )
            zf.writestr(
                "OEBPS/content.opf",
                """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
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
</body>
</html>""",
            )

        state_dir = str(tmp / "state")
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"polish": True, "review": False, "review_autofix": False},
                "paths": {"state_dir": state_dir},
            }
        )

        # Translation preserves markers, but polish strips them
        def polish_breaks_handler(messages, tier, json_mode):
            system = messages[0]["content"]
            user = messages[-1]["content"]

            if "translator" in system.lower():
                # Translation: preserve markers
                paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE)
                if paragraphs:
                    translations = []
                    for idx, text in paragraphs:
                        marker_pattern = r"(\[\[[A-Z_]+_(START|END)\]\])"
                        parts = re.split(marker_pattern, text)
                        result = []
                        for part in parts:
                            if re.match(marker_pattern, part):
                                result.append(part)
                            elif part.strip() and part not in ("START", "END"):
                                result.append(f"中文{idx}")
                        translations.append("".join(result))
                    return json.dumps({"translations": translations}, ensure_ascii=False)

            elif "Polish" in user or "polish" in system.lower():
                # Polish: strip all markers
                for msg in reversed(messages[:-1]):
                    if msg.get("role") == "assistant":
                        try:
                            prev = json.loads(msg["content"])
                            trans = prev.get("translations", [])
                            if trans:
                                polished = [
                                    re.sub(r"\[\[[A-Z_]+_(START|END)\]\]", "", t) for t in trans
                                ]
                                return json.dumps({"polished": polished}, ensure_ascii=False)
                        except Exception:
                            pass

            return base_routing_handler(messages, tier, json_mode)

        client = FakeClient(handler=polish_breaks_handler)
        orch = Orchestrator(config, client=client)

        # Should complete but mark preservation_failed
        store = orch.run(epub_path)

        # Verify preservation_failed is set
        ch = store.load_chapter(0)
        emphasis_seg = next((s for s in ch.segments if s.source and "emphasis" in s.source), None)
        assert emphasis_seg is not None
        target_inline = emphasis_seg.meta.get("target_inline", {})
        assert target_inline.get("preservation_failed") == True, (
            "Polish removing markers must set preservation_failed"
        )

        # Export must fail
        md_path = str(tmp / "output.md")
        with pytest.raises(InlinePreservationError):
            orch.run_assemble(epub_path, out_path=md_path, out_format="markdown")

        assert not Path(md_path).exists(), "Failed export must not create output file"


def test_legacy_state_missing_target_inline():
    """Legacy state without target_inline rejects export fail-closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create EPUB with emphasis
        epub_path = str(tmp / "test.epub")
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebeps-package+xml"/></rootfiles></container>',
            )
            zf.writestr(
                "OEBPS/content.opf",
                """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
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
  <p>Text with <em>emphasis</em> word.</p>
</body>
</html>""",
            )

        state_dir = str(tmp / "state")
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"polish": False, "review": False, "review_autofix": False},
                "paths": {"state_dir": state_dir},
            }
        )

        client = FakeClient(handler=base_routing_handler)
        orch = Orchestrator(config, client=client)

        # First, run normal translation to create valid state
        store = orch.run(epub_path)

        # Now simulate legacy state: remove target_inline from segment with emphasis
        ch = store.load_chapter(0)
        for seg in ch.segments:
            if seg.source and "emphasis" in seg.source:
                # Remove target_inline to simulate legacy state
                if "target_inline" in seg.meta:
                    del seg.meta["target_inline"]

        store.save_chapter(ch)

        # Attempt export - must fail due to missing target_inline
        epub_out = str(tmp / "bad.epub")

        with pytest.raises(InlinePreservationError):
            orch.run_assemble(epub_path, out_path=epub_out, out_format="epub")

        # Verify no output file
        assert not Path(epub_out).exists(), "Legacy state export must not create output"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
