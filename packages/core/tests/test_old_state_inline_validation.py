"""Test old state (v1166-era) inline validation catches missing emphasis preservation."""

import json
import tempfile
from pathlib import Path

import pytest

from wenyi_core.assemble.inline_validation import InlinePreservationError
from wenyi_core.config import Config
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.file import FileStorage


def test_old_state_with_emphasis_fails_export():
    """Old state (no INLINE_META_KEY, no target_inline) but source has emphasis → export fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = tmp / "test.epub"

        # Create EPUB with emphasis markup
        import zipfile

        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
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
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
  </spine>
</package>""",
            )
            # HTML with emphasis tags
            zf.writestr(
                "OEBPS/chapter1.xhtml",
                """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
<p>This paragraph has <em>emphasis</em> and <strong>strong</strong> text.</p>
<p>Another paragraph with <i>italic</i> words.</p>
</body>
</html>""",
            )

        state_dir = tmp / "state"

        from wenyi_core.llm.providers.fake import FakeClient
        from tests.fake_llm import routing_handler

        config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {
                    "polish": False,
                    "review": False,
                    "review_autofix": False,
                    "book_understanding": False,
                },
                "paths": {"state_dir": str(state_dir)},
            }
        )

        client = FakeClient(handler=routing_handler)
        orch = Orchestrator(config, client=client)

        # Run translation to get state with inline metadata
        store = orch.run(str(epub_path))
        chapter = store.load_chapter(0)

        # Simulate old state by removing inline metadata (both old and new keys)
        for seg in chapter.segments:
            # Remove new-style keys
            if "epub_inline" in seg.meta:
                del seg.meta["epub_inline"]
            if "target_inline" in seg.meta:
                del seg.meta["target_inline"]
            # Also remove old names if present
            if "source_inline" in seg.meta:
                del seg.meta["source_inline"]
        store.save_chapter(chapter)

        # Now try to export - should fail with InlinePreservationError
        md_out_path = str(tmp / "output.md")
        epub_out_path = str(tmp / "output.epub")

        # Test Markdown export
        with pytest.raises(InlinePreservationError) as exc_info:
            orch.run_assemble(str(epub_path), out_path=md_out_path, out_format="markdown")
        
        error_msg = str(exc_info.value)
        assert "emphasis markup" in error_msg.lower()
        assert "old version" in error_msg.lower() or "retranslate" in error_msg.lower()
        assert not Path(md_out_path).exists(), "MD output should not exist after validation failure"

        # Test EPUB export
        with pytest.raises(InlinePreservationError) as exc_info:
            orch.run_assemble(str(epub_path), out_path=epub_out_path, out_format="epub")
        
        error_msg = str(exc_info.value)
        assert "emphasis markup" in error_msg.lower()
        assert not Path(epub_out_path).exists(), "EPUB output should not exist after validation failure"


def test_new_state_with_inline_metadata_succeeds():
    """New state (has INLINE_META_KEY and target_inline) → export succeeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = tmp / "test.epub"

        # Create EPUB with emphasis markup (same as first test)
        import zipfile

        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
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
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
  </spine>
</package>""",
            )
            zf.writestr(
                "OEBPS/chapter1.xhtml",
                """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
<p>This paragraph has <em>emphasis</em> and <strong>strong</strong> text.</p>
<p>Another paragraph with <i>italic</i> words.</p>
</body>
</html>""",
            )

        state_dir = tmp / "state"

        from wenyi_core.llm.providers.fake import FakeClient
        from tests.fake_llm import routing_handler

        config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {
                    "polish": False,
                    "review": False,
                    "review_autofix": False,
                    "book_understanding": False,
                },
                "paths": {"state_dir": str(state_dir)},
            }
        )

        client = FakeClient(handler=routing_handler)
        orch = Orchestrator(config, client=client)

        # Run translation
        store = orch.run(str(epub_path))

        # Verify state has inline metadata
        chapter = store.load_chapter(0)
        has_inline = any(
            seg.meta.get("epub_inline") and seg.meta.get("target_inline")
            for seg in chapter.segments
        )
        assert has_inline, "New state should have inline metadata"

        # Export should succeed
        md_path = str(tmp / "output.md")
        epub_out_path = str(tmp / "output.epub")

        # Both exports should succeed without raising
        orch.run_assemble(str(epub_path), out_path=md_path, out_format="markdown")
        assert Path(md_path).exists(), "MD output should exist"

        orch.run_assemble(str(epub_path), out_path=epub_out_path, out_format="epub")
        assert Path(epub_out_path).exists(), "EPUB output should exist"


def test_old_state_without_emphasis_is_compatible():
    """Old state without emphasis in source → export succeeds (backward compatible)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = tmp / "test.epub"

        # Create simple EPUB without any emphasis markup
        import zipfile

        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
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
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
  </spine>
</package>""",
            )
            # Plain text, no emphasis tags
            zf.writestr(
                "OEBPS/chapter1.xhtml",
                """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
<p>Plain paragraph without any emphasis.</p>
<p>Another plain paragraph.</p>
</body>
</html>""",
            )

        state_dir = tmp / "state"

        from wenyi_core.llm.providers.fake import FakeClient
        from tests.fake_llm import routing_handler

        config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {
                    "polish": False,
                    "review": False,
                    "review_autofix": False,
                    "book_understanding": False,
                },
                "paths": {"state_dir": str(state_dir)},
            }
        )

        client = FakeClient(handler=routing_handler)
        orch = Orchestrator(config, client=client)

        # Run translation
        store = orch.run(str(epub_path))

        # Simulate old state by removing inline metadata
        chapter = store.load_chapter(0)
        for seg in chapter.segments:
            if "epub_inline" in seg.meta:
                del seg.meta["epub_inline"]
            if "target_inline" in seg.meta:
                del seg.meta["target_inline"]
        store.save_chapter(chapter)

        # Export should succeed (no emphasis to lose)
        md_path = str(tmp / "output.md")
        orch.run_assemble(str(epub_path), out_path=md_path, out_format="markdown")
        assert Path(md_path).exists(), "MD output should exist for plain text old state"


def test_error_message_mentions_old_state_issue():
    """Error message clearly explains old state inline preservation failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = tmp / "test.epub"

        # Create EPUB with emphasis
        import zipfile

        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
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
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test</dc:title>
    <dc:language>en</dc:language>
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
<body><p>Text with <em>emphasis</em>.</p></body>
</html>""",
            )

        state_dir = tmp / "state"

        from wenyi_core.llm.providers.fake import FakeClient
        from tests.fake_llm import routing_handler

        config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {
                    "polish": False,
                    "review": False,
                    "review_autofix": False,
                    "book_understanding": False,
                },
                "paths": {"state_dir": str(state_dir)},
            }
        )

        client = FakeClient(handler=routing_handler)
        orch = Orchestrator(config, client=client)
        store = orch.run(str(epub_path))

        # Strip inline metadata
        chapter = store.load_chapter(0)
        for seg in chapter.segments:
            if "epub_inline" in seg.meta:
                del seg.meta["epub_inline"]
            if "target_inline" in seg.meta:
                del seg.meta["target_inline"]
        store.save_chapter(chapter)

        # Try to export - catch error and verify message
        md_path = str(tmp / "output.md")
        with pytest.raises(InlinePreservationError) as exc_info:
            orch.run_assemble(str(epub_path), out_path=md_path, out_format="markdown")

        error_msg = str(exc_info.value)
        
        # Error should mention emphasis/inline markup
        assert "emphasis" in error_msg.lower() or "inline" in error_msg.lower()
        
        # Error should mention old version or retranslation
        assert "old version" in error_msg.lower() or "retranslate" in error_msg.lower()
        
        # Error should identify specific chapter/segment
        assert "chapter" in error_msg.lower() and "segment" in error_msg.lower()
        
        # Should NOT create output file
        assert not Path(md_path).exists(), "Output should not be created on validation failure"
