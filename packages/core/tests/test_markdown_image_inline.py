"""Test Markdown image extraction and inline markup preservation."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from wenyi_core.assemble.inline_detection import detect_unpreservable_inline, format_inline_report
from wenyi_core.assemble.text_writer import (
    _extract_images_from_epub,
    _extract_segment_images_from_epub,
    _normalize_epub_href,
    _safe_filename,
)
from wenyi_core.markup.inline_markers import (
    InlineNode,
    decode_inline_markers,
    encode_inline_markers,
)
from wenyi_core.models.document import Chapter, Segment


class TestSafeFilename:
    """Test filename sanitization."""

    def test_basic_sanitization(self):
        assert _safe_filename("hello world", "fallback") == "hello world"
        assert _safe_filename("file<>name", "fallback") == "file__name"
        assert _safe_filename("path/to\\file", "fallback") == "path_to_file"

    def test_empty_and_dots(self):
        assert _safe_filename("", "fallback") == "fallback"
        assert _safe_filename("...", "fallback") == "fallback"
        assert _safe_filename("  . ", "fallback") == "fallback"

    def test_control_characters(self):
        assert _safe_filename("file\x00name", "fallback") == "file_name"
        assert _safe_filename("file\x1fname", "fallback") == "file_name"


class TestNormalizeEpubHref:
    """Test EPUB href normalization."""

    def test_absolute_paths(self):
        assert _normalize_epub_href("OEBPS/text.xhtml", "/images/cover.jpg") == "images/cover.jpg"

    def test_relative_paths(self):
        assert _normalize_epub_href("OEBPS/text.xhtml", "images/img.jpg") == "OEBPS/images/img.jpg"
        assert _normalize_epub_href("OEBPS/chapter1.xhtml", "../images/img.jpg") == "images/img.jpg"

    def test_current_directory(self):
        assert _normalize_epub_href("text.xhtml", "./img.jpg") == "img.jpg"
        assert _normalize_epub_href("text.xhtml", "img.jpg") == "img.jpg"


class TestImageExtraction:
    """Test image extraction from EPUB."""

    def _create_test_epub(self, tmp_path: Path) -> Path:
        """Create a minimal EPUB with images."""
        epub_path = tmp_path / "test.epub"
        
        with zipfile.ZipFile(epub_path, 'w') as zf:
            # Add mimetype
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            
            # Add container.xml
            zf.writestr("META-INF/container.xml", """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""")
            
            # Add images
            zf.writestr("OEBPS/images/cover.jpg", b"fake_jpg_data")
            zf.writestr("OEBPS/images/diagram.png", b"fake_png_data")
            zf.writestr("OEBPS/styles/style.css", "body { margin: 0; }")
        
        return epub_path

    def test_extract_images(self, tmp_path):
        """Test basic image extraction."""
        epub_path = self._create_test_epub(tmp_path)
        assets_dir = tmp_path / "assets"
        
        image_map = _extract_images_from_epub(str(epub_path), str(assets_dir))
        
        assert len(image_map) == 2
        assert "OEBPS/images/cover.jpg" in image_map
        assert "OEBPS/images/diagram.png" in image_map
        assert image_map["OEBPS/images/cover.jpg"] == "assets/cover.jpg"
        assert image_map["OEBPS/images/diagram.png"] == "assets/diagram.png"
        
        # Verify files exist
        assert (assets_dir / "cover.jpg").exists()
        assert (assets_dir / "diagram.png").exists()

    def test_nonexistent_epub(self, tmp_path):
        """Test handling of nonexistent EPUB."""
        image_map = _extract_images_from_epub("/nonexistent.epub", str(tmp_path / "assets"))
        assert image_map == {}

    def test_duplicate_image_names(self, tmp_path):
        """Test handling of duplicate image names in different directories."""
        epub_path = tmp_path / "test.epub"
        
        with zipfile.ZipFile(epub_path, 'w') as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr("images/img.jpg", b"data1")
            zf.writestr("other/img.jpg", b"data2")
        
        assets_dir = tmp_path / "assets"
        image_map = _extract_images_from_epub(str(epub_path), str(assets_dir))
        
        assert len(image_map) == 2
        assert len(set(image_map.values())) == 2  # Both mapped to different names
        
        # Check files exist and have correct content
        extracted_files = list((assets_dir).glob("*.jpg"))
        assert len(extracted_files) == 2


class TestInlineMarkers:
    """Test inline markup marker encoding/decoding."""

    def test_simple_italic(self):
        """Test encoding and decoding simple italic."""
        text = "The Force is strong"
        nodes = [
            {"tag": "i", "offset": 4, "placement": "inline", "id": "test_0"}
        ]
        
        encoded, inline_nodes = encode_inline_markers(text, nodes, len(text))
        
        assert "[[I_START]]" in encoded
        assert "[[I_END]]" in encoded
        assert len(inline_nodes) == 1
        
        # Decode
        decoded, ranges, warnings = decode_inline_markers(encoded, inline_nodes)
        
        assert decoded == text
        assert len(ranges) == 1
        assert ranges[0][2] == "i"
        assert len(warnings) == 0

    def test_nested_emphasis(self):
        """Test nested <strong> inside <em>."""
        text = "This is very important text"
        nodes = [
            {"tag": "em", "offset": 8, "placement": "inline", "id": "em_0"},
            {"tag": "strong", "offset": 13, "placement": "inline", "id": "strong_0"},
        ]
        
        encoded, inline_nodes = encode_inline_markers(text, nodes, len(text))
        
        assert "[[EM_START]]" in encoded
        assert "[[STRONG_START]]" in encoded
        assert len(inline_nodes) == 2

    def test_missing_end_marker(self):
        """Test detection of missing end marker."""
        original_nodes = [InlineNode(tag="i", start=0, end=5, marker_name="I")]
        translated = "Text with [[I_START]] but no end"
        
        decoded, ranges, warnings = decode_inline_markers(translated, original_nodes)
        
        assert len(warnings) > 0
        assert any("Unclosed" in w for w in warnings)

    def test_mismatched_markers(self):
        """Test detection of mismatched markers."""
        original_nodes = [
            InlineNode(tag="i", start=0, end=5, marker_name="I"),
            InlineNode(tag="strong", start=6, end=12, marker_name="STRONG"),
        ]
        translated = "Text [[I_START]] italic [[STRONG_END]] [[STRONG_START]] bold [[I_END]]"
        
        decoded, ranges, warnings = decode_inline_markers(translated, original_nodes)
        
        assert len(warnings) > 0
        assert any("Mismatched" in w or "Unmatched" in w for w in warnings)

    def test_count_mismatch(self):
        """Test warning when inline node count changes."""
        original_nodes = [
            InlineNode(tag="i", start=0, end=5, marker_name="I"),
            InlineNode(tag="strong", start=6, end=12, marker_name="STRONG"),
        ]
        # Translation has only one pair
        translated = "Text [[I_START]]italic[[I_END]]"
        
        decoded, ranges, warnings = decode_inline_markers(translated, original_nodes)
        
        assert len(warnings) > 0
        assert any("count mismatch" in w.lower() for w in warnings)


class TestInlineDetection:
    """Test inline markup preservation detection."""

    def test_detect_untranslated_inline(self):
        """Test detection of untranslated segments with inline markup."""
        seg = Segment(
            index=0,
            source="The Force",
            target=None,
            meta={
                "epub_inline": {
                    "version": 1,
                    "source_length": 9,
                    "nodes": [
                        {"tag": "i", "offset": 4, "placement": "inline"}
                    ]
                }
            }
        )
        
        ch = Chapter(index=0, title="Test", segments=[seg])
        
        issues = detect_unpreservable_inline([ch])
        
        assert 0 in issues
        assert len(issues[0]) == 1
        assert issues[0][0]["reason"] == "untranslated"
        assert issues[0][0]["inline_node_count"] == 1

    def test_detect_proportional_mapping(self):
        """Test detection of segments with proportional mapping concerns."""
        seg = Segment(
            index=0,
            source="The Force",
            target="原力",
            meta={
                "epub_inline": {
                    "version": 1,
                    "source_length": 9,
                    "nodes": [
                        {"tag": "i", "offset": 4, "placement": "inline"}
                    ]
                }
            }
        )
        
        ch = Chapter(index=0, title="Test", segments=[seg])
        
        issues = detect_unpreservable_inline([ch])
        
        assert 0 in issues
        assert len(issues[0]) == 1
        assert issues[0][0]["reason"] == "proportional_mapping_unreliable"

    def test_no_issues(self):
        """Test chapters without inline markup issues."""
        seg = Segment(index=0, source="Plain text", target="纯文本", meta={})
        ch = Chapter(index=0, title="Test", segments=[seg])
        
        issues = detect_unpreservable_inline([ch])
        
        assert len(issues) == 0

    def test_format_report(self):
        """Test report formatting."""
        seg = Segment(
            index=5,
            source="The Force is strong with this one",
            target=None,
            meta={
                "epub_inline": {
                    "version": 1,
                    "source_length": 33,
                    "nodes": [
                        {"tag": "i", "id": "inline_0"},
                        {"tag": "strong", "id": "inline_1"}
                    ]
                }
            }
        )
        
        ch = Chapter(index=2, title="Test", segments=[seg])
        
        issues = detect_unpreservable_inline([ch])
        report = format_inline_report(issues)
        
        assert "Chapter 2" in report
        assert "Segment 5" in report
        assert "2 inline nodes" in report
        assert "<i>" in report
        assert "<strong>" in report
        assert "untranslated" in report
