"""Test Markdown image positioning and chapter titles."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from wenyi_core.assemble.writer import assemble
from wenyi_core.markup.contracts import INLINE_ID_ATTR, INLINE_META_KEY
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.storage.file import FileStorage


def _create_test_epub_with_images(
    epub_path: str,
    images_data: dict[str, bytes],
    html_resources: dict[str, str],
) -> None:
    """Create a minimal EPUB with images for testing."""
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Minimal EPUB structure
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )

        # Package document
        opf_content = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
"""
        # Add HTML resources to manifest
        for idx, href in enumerate(html_resources.keys()):
            opf_content += (
                f'    <item id="html{idx}" href="{href}" media-type="application/xhtml+xml"/>\n'
            )
        # Add images to manifest
        for idx, img_name in enumerate(images_data.keys()):
            ext = img_name.split(".")[-1].lower()
            media_type = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif") else "image/png"
            opf_content += (
                f'    <item id="img{idx}" href="{img_name}" media-type="{media_type}"/>\n'
            )
        opf_content += """  </manifest>
  <spine>
"""
        for idx in range(len(html_resources)):
            opf_content += f'    <itemref idref="html{idx}"/>\n'
        opf_content += """  </spine>
</package>"""
        zf.writestr("content.opf", opf_content)

        # Write HTML resources
        for href, content in html_resources.items():
            zf.writestr(href, content)

        # Write image files
        for img_name, img_data in images_data.items():
            zf.writestr(img_name, img_data)


def test_markdown_image_order_with_chapter_title():
    """Test that images and paragraphs maintain document order with chapter title."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        store = FileStorage(str(tmp / "book"))

        # Create EPUB with images
        images = {
            "images/cover.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
            "images/diagram.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
        }
        html_resources = {
            "chapter1.xhtml": f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
  <img src="images/cover.png" alt="Cover Image" {INLINE_ID_ATTR}="img_cover"/>
  <p {INLINE_ID_ATTR}="p1">First paragraph text.</p>
  <img src="images/diagram.png" alt="Diagram" {INLINE_ID_ATTR}="img_diagram"/>
  <p {INLINE_ID_ATTR}="p2">Second paragraph text.</p>
</body>
</html>""",
        }
        _create_test_epub_with_images(epub_path, images, html_resources)

        # Create segments with image metadata
        seg1 = Segment(
            index=0,
            source="First paragraph text.",
            target="第一段文字。",
            resource_href="chapter1.xhtml",
            meta={
                INLINE_META_KEY: {
                    "version": 1,
                    "source_length": 21,
                    "nodes": [
                        {"id": "img_cover", "tag": "img", "placement": "block"},
                        {"id": "p1", "tag": "p", "placement": "block"},
                    ],
                }
            },
        )

        seg2 = Segment(
            index=1,
            source="Second paragraph text.",
            target="第二段文字。",
            resource_href="chapter1.xhtml",
            meta={
                INLINE_META_KEY: {
                    "version": 1,
                    "source_length": 22,
                    "nodes": [
                        {"id": "img_diagram", "tag": "img", "placement": "block"},
                        {"id": "p2", "tag": "p", "placement": "block"},
                    ],
                }
            },
        )

        chapter = Chapter(index=0, title="第一章：开始", segments=[seg1, seg2])
        store.save_chapter(chapter)

        manifest = {
            "fmt": "epub",
            "source_lang": "en",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapter_count": 1,
            "source_sha256": "test123",
            "chapters": [{"index": 0, "title": "第一章：开始", "status": "done"}],
            "meta": {
                "epub_resources": [
                    {"href": "chapter1.xhtml", "media_type": "application/xhtml+xml"}
                ]
            },
        }
        store.save_manifest(manifest)

        # Assemble markdown
        out_md = str(tmp / "output.md")
        assemble(store, epub_path, out_path=out_md, out_format="markdown")

        # Verify output
        md_content = Path(out_md).read_text(encoding="utf-8")
        lines = [line.strip() for line in md_content.strip().split("\n") if line.strip()]

        # Expected order: chapter title, cover image, paragraph 1, diagram image, paragraph 2
        assert lines[0] == "# 第一章：开始", "Chapter title should be first"
        assert lines[1].startswith("![Cover Image]"), "Cover image should be second"
        assert lines[1].endswith("(assets/cover.png)"), "Image path should be relative"
        assert "第一段文字" in lines[2], "First paragraph should be after cover image"
        assert lines[3].startswith("![Diagram]"), "Diagram image should be after first paragraph"
        assert lines[3].endswith("(assets/diagram.png)"), "Diagram path should be relative"
        assert "第二段文字" in lines[4], "Second paragraph should be after diagram"

        # Verify assets directory
        assets_dir = tmp / "assets"
        assert assets_dir.exists(), "Assets directory should exist"
        assert (assets_dir / "cover.png").exists(), "Cover image should be extracted"
        assert (assets_dir / "diagram.png").exists(), "Diagram image should be extracted"


def test_markdown_rejects_inline_images():
    """Test that inline images (within text) are rejected with clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        store = FileStorage(str(tmp / "book"))

        # Create EPUB with inline image
        images = {"icon.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 100}
        html_resources = {
            "chapter1.xhtml": f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
  <p {INLINE_ID_ATTR}="p1">Text with <img src="icon.png" alt="icon" {INLINE_ID_ATTR}="inline_icon"/> inside.</p>
</body>
</html>""",
        }
        _create_test_epub_with_images(epub_path, images, html_resources)

        # Create segment with inline image metadata
        seg = Segment(
            index=0,
            source="Text with icon inside.",
            target="文字中包含图标。",
            resource_href="chapter1.xhtml",
            meta={
                INLINE_META_KEY: {
                    "version": 1,
                    "source_length": 23,
                    "nodes": [
                        {"id": "inline_icon", "tag": "img", "placement": "inline"},
                    ],
                }
            },
        )

        chapter = Chapter(index=0, title="Chapter 1", segments=[seg])
        store.save_chapter(chapter)

        manifest = {
            "fmt": "epub",
            "source_lang": "en",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapter_count": 1,
            "source_sha256": "test123",
            "chapters": [{"index": 0, "title": "Chapter 1", "status": "done"}],
            "meta": {"epub_resources": [{"href": "chapter1.xhtml"}]},
        }
        store.save_manifest(manifest)

        # Assemble should fail with clear error
        out_md = str(tmp / "output.md")
        with pytest.raises(ValueError) as exc_info:
            assemble(store, epub_path, out_path=out_md, out_format="markdown")

        error_msg = str(exc_info.value)
        assert "Chapter 0, Segment 0" in error_msg
        assert "Inline images" in error_msg
        assert "not supported in Markdown" in error_msg


def test_markdown_image_duplicate_names():
    """Test that duplicate image names are handled safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        store = FileStorage(str(tmp / "book"))

        # Create EPUB with duplicate image names in different directories
        images = {
            "images/fig1/image.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
            "images/fig2/image.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
        }
        html_resources = {
            "chapter1.xhtml": f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
  <img src="images/fig1/image.png" alt="Fig 1" {INLINE_ID_ATTR}="img1"/>
  <p {INLINE_ID_ATTR}="p1">Text.</p>
  <img src="images/fig2/image.png" alt="Fig 2" {INLINE_ID_ATTR}="img2"/>
</body>
</html>""",
        }
        _create_test_epub_with_images(epub_path, images, html_resources)

        seg = Segment(
            index=0,
            source="Text.",
            target="文字。",
            resource_href="chapter1.xhtml",
            meta={
                INLINE_META_KEY: {
                    "version": 1,
                    "source_length": 5,
                    "nodes": [
                        {"id": "img1", "tag": "img", "placement": "block"},
                        {"id": "p1", "tag": "p", "placement": "block"},
                        {"id": "img2", "tag": "img", "placement": "block"},
                    ],
                }
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
            "source_sha256": "test123",
            "chapters": [{"index": 0, "title": "Test", "status": "done"}],
            "meta": {"epub_resources": [{"href": "chapter1.xhtml"}]},
        }
        store.save_manifest(manifest)

        # Assemble markdown
        out_md = str(tmp / "output.md")
        assemble(store, epub_path, out_path=out_md, out_format="markdown")

        # Verify both images exist with different names
        assets_dir = tmp / "assets"
        image_files = list(assets_dir.glob("*.png"))
        assert len(image_files) == 2, "Should have 2 images with unique names"

        # Verify output references both images
        md_content = Path(out_md).read_text(encoding="utf-8")
        assert "![Fig 1]" in md_content
        assert "![Fig 2]" in md_content
        assert md_content.count("(assets/image") == 2, "Should reference 2 images"


def test_markdown_missing_image_files():
    """Test that missing image files cause explicit failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        store = FileStorage(str(tmp / "book"))

        # Create EPUB with referenced but missing image files
        html_resources = {
            "chapter1.xhtml": f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
  <div><img src="missing.png" alt="Missing"/></div>
  <p>Text.</p>
</body>
</html>""",
        }
        _create_test_epub_with_images(epub_path, {}, html_resources)

        # Ingest EPUB
        from wenyi_core.ingest.epub_reader import read_epub
        doc = read_epub(epub_path, source_lang="en", target_lang="zh")
        
        # Translate segments
        for seg in doc.chapters[0].segments:
            seg.target = "文字。"

        chapter = doc.chapters[0]
        store.save_chapter(chapter)

        manifest = {
            "fmt": "epub",
            "source_lang": "en",
            "target_lang": "zh",
            "source_path": epub_path,
            "chapter_count": 1,
            "source_sha256": "test123",
            "chapters": [{"index": 0, "title": "Test", "status": "done"}],
            "meta": doc.meta,
        }
        store.save_manifest(manifest)

        # Assemble should fail with FileNotFoundError
        out_md = str(tmp / "output.md")
        with pytest.raises(FileNotFoundError) as exc_info:
            assemble(store, epub_path, out_path=out_md, out_format="markdown")
        
        error_msg = str(exc_info.value)
        assert "missing.png" in error_msg
        assert "Required image files missing" in error_msg
        # Output file should NOT exist
        assert not Path(out_md).exists()
