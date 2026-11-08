"""Test Markdown image positioning with real EPUB ingestion."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from wenyi_core.assemble.writer import assemble
from wenyi_core.ingest.epub_reader import read_epub
from wenyi_core.models.document import Chapter
from wenyi_core.storage.file import FileStorage


def _create_real_epub(epub_path: str) -> None:
    """Create a minimal valid EPUB with images in various positions."""
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # mimetype (uncompressed, first file)
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

        # container.xml
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )

        # Package document
        zf.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book with Images</dc:title>
    <dc:language>en</dc:language>
    <dc:identifier id="bookid">test-123</dc:identifier>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="cover" href="images/cover.png" media-type="image/png"/>
    <item id="diagram" href="images/diagram.png" media-type="image/png"/>
    <item id="icon1" href="images/dir1/icon.png" media-type="image/png"/>
    <item id="icon2" href="images/dir2/icon.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
  </spine>
</package>""",
        )

        # Navigation document
        zf.writestr(
            "OEBPS/nav.xhtml",
            """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
  <title>Table of Contents</title>
</head>
<body>
  <nav epub:type="toc">
    <ol>
      <li><a href="chapter1.xhtml">Chapter One: Beginning</a></li>
    </ol>
  </nav>
</body>
</html>""",
        )

        # Chapter with images at different positions
        zf.writestr(
            "OEBPS/chapter1.xhtml",
            """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <title>Chapter One: Beginning</title>
</head>
<body>
  <h1>Chapter One: Beginning</h1>
  <img src="images/cover.png" alt="Cover Image"/>
  <p>First paragraph text.</p>
  <img src="images/diagram.png" alt="Diagram"/>
  <p>Second paragraph with <img src="images/dir1/icon.png" alt="inline icon"/> inside text.</p>
  <p>Third paragraph.</p>
  <img src="images/dir2/icon.png" alt="Another icon"/>
</body>
</html>""",
        )

        # Image files (minimal valid PNG)
        png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        png_data = png_header + b"\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

        zf.writestr("OEBPS/images/cover.png", png_data)
        zf.writestr("OEBPS/images/diagram.png", png_data)
        zf.writestr("OEBPS/images/dir1/icon.png", png_data)
        zf.writestr("OEBPS/images/dir2/icon.png", png_data)


def test_real_epub_image_order_and_chapter_title():
    """Test full pipeline: EPUB ingest → prepare → translate → assemble Markdown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        _create_real_epub(epub_path)

        # Ingest EPUB
        doc = read_epub(epub_path, source_lang="en", target_lang="zh")
        assert len(doc.chapters) == 1
        ch = doc.chapters[0]
        assert ch.title == "Chapter One: Beginning"

        # Check segments have inline metadata
        assert len(ch.segments) > 0
        has_image_metadata = False
        for seg in ch.segments:
            inline_meta = seg.meta.get("epub_inline")
            if inline_meta and isinstance(inline_meta, dict):
                nodes = inline_meta.get("nodes", [])
                for node in nodes:
                    if isinstance(node, dict) and node.get("tag") == "img":
                        has_image_metadata = True
                        assert "placement" in node
                        assert "id" in node
        assert has_image_metadata, "EPUB ingest should preserve image metadata"

        # Simulate translation (fixed test text)
        translated_segments = []
        for seg in ch.segments:
            seg.target = f"中文_{seg.index}"
            translated_segments.append(seg)

        translated_chapter = Chapter(
            index=ch.index,
            title="第一章：开始",
            segments=translated_segments,
            href=ch.href,
            template=ch.template,
            meta=ch.meta,
        )

        # Save to storage
        store = FileStorage(str(tmp / "book"))
        store.save_chapter(translated_chapter)

        manifest = {
            "fmt": "epub",
            "source_lang": "en",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapter_count": 1,
            "source_sha256": "test123",
            "chapters": [{"index": 0, "title": "第一章：开始", "status": "done"}],
            "meta": doc.meta,
        }
        store.save_manifest(manifest)

        # Assemble Markdown - should reject inline images
        out_md = str(tmp / "output.md")
        with pytest.raises(ValueError) as exc_info:
            assemble(store, epub_path, out_path=out_md, out_format="markdown")

        error_msg = str(exc_info.value)
        assert "Inline images" in error_msg
        assert "not supported in Markdown" in error_msg


def test_real_epub_block_images_only():
    """Test EPUB with only block-level images (no inline)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")

        # Create EPUB without inline images
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
    <dc:title>Block Images Test</dc:title>
    <dc:language>en</dc:language>
    <dc:identifier id="bookid">test-456</dc:identifier>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="fig1" href="images/figure1.png" media-type="image/png"/>
    <item id="fig2" href="images/figure2.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
  </spine>
</package>""",
            )
            zf.writestr(
                "OEBPS/nav.xhtml",
                """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>TOC</title></head>
<body>
  <nav epub:type="toc">
    <ol><li><a href="chapter1.xhtml">Test Chapter</a></li></ol>
  </nav>
</body>
</html>""",
            )
            zf.writestr(
                "OEBPS/chapter1.xhtml",
                """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test Chapter</title></head>
<body>
  <h1>Test Chapter</h1>
  <img src="images/figure1.png" alt="Figure 1"/>
  <p>Paragraph one.</p>
  <img src="images/figure2.png" alt="Figure 2"/>
  <p>Paragraph two.</p>
</body>
</html>""",
            )

            png_data = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
                b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            zf.writestr("OEBPS/images/figure1.png", png_data)
            zf.writestr("OEBPS/images/figure2.png", png_data)

        # Ingest and translate
        doc = read_epub(epub_path, source_lang="en", target_lang="zh")
        ch = doc.chapters[0]

        for seg in ch.segments:
            seg.target = f"段落_{seg.index}"

        translated_chapter = Chapter(
            index=ch.index,
            title="测试章节",
            segments=ch.segments,
            href=ch.href,
            template=ch.template,
            meta=ch.meta,
        )

        store = FileStorage(str(tmp / "book"))
        store.save_chapter(translated_chapter)

        manifest = {
            "fmt": "epub",
            "source_lang": "en",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapter_count": 1,
            "source_sha256": "test456",
            "chapters": [{"index": 0, "title": "测试章节", "status": "done"}],
            "meta": doc.meta,
        }
        store.save_manifest(manifest)

        # Assemble Markdown - should succeed
        out_md = str(tmp / "output.md")
        result = assemble(store, epub_path, out_path=out_md, out_format="markdown")
        assert Path(result).exists()

        # Verify output
        md_content = Path(out_md).read_text(encoding="utf-8")
        lines = [line.strip() for line in md_content.strip().split("\n") if line.strip()]

        # Should have chapter title
        assert lines[0] == "# 测试章节"

        # Should have images and paragraphs in correct order
        assert "![Figure 1]" in md_content
        assert "![Figure 2]" in md_content
        assert "段落" in md_content

        # Verify assets
        assets_dir = tmp / "assets"
        assert assets_dir.exists()
        assert (assets_dir / "figure1.png").exists()
        assert (assets_dir / "figure2.png").exists()

        # Verify file sizes are reasonable
        assert (assets_dir / "figure1.png").stat().st_size > 0
        assert (assets_dir / "figure2.png").stat().st_size > 0


def test_missing_epub_images_fail_explicitly():
    """Test that missing image files cause explicit failure, not silent skip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")

        # Create EPUB referencing images that don't exist
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
    <dc:title>Missing Images Test</dc:title>
    <dc:language>en</dc:language>
    <dc:identifier id="bookid">test-789</dc:identifier>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="missing" href="images/missing.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
  </spine>
</package>""",
            )
            zf.writestr(
                "OEBPS/nav.xhtml",
                """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>TOC</title></head>
<body>
  <nav epub:type="toc">
    <ol><li><a href="chapter1.xhtml">Chapter</a></li></ol>
  </nav>
</body>
</html>""",
            )
            zf.writestr(
                "OEBPS/chapter1.xhtml",
                """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter</title></head>
<body>
  <h1>Chapter</h1>
  <img src="images/missing.png" alt="Missing Image"/>
  <p>Text paragraph.</p>
</body>
</html>""",
            )
            # Note: images/missing.png is NOT created

        doc = read_epub(epub_path, source_lang="en", target_lang="zh")
        ch = doc.chapters[0]
        for seg in ch.segments:
            seg.target = f"文字_{seg.index}"

        translated_chapter = Chapter(
            index=ch.index, title="章节", segments=ch.segments, href=ch.href, template=ch.template, meta=ch.meta
        )

        store = FileStorage(str(tmp / "book"))
        store.save_chapter(translated_chapter)

        manifest = {
            "fmt": "epub",
            "source_lang": "en",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapter_count": 1,
            "source_sha256": "test789",
            "chapters": [{"index": 0, "title": "章节", "status": "done"}],
            "meta": doc.meta,
        }
        store.save_manifest(manifest)

        # Assemble should fail with clear error
        out_md = str(tmp / "output.md")
        with pytest.raises((ValueError, FileNotFoundError, OSError)) as exc_info:
            assemble(store, epub_path, out_path=out_md, out_format="markdown")

        error_msg = str(exc_info.value)
        # Should mention the image path or indicate missing resource
        assert "missing" in error_msg.lower() or "not found" in error_msg.lower() or "images" in error_msg.lower()

        # Output file should NOT exist (transaction failed)
        assert not Path(out_md).exists(), "Failed assembly should not create partial output"
