"""Test fail-closed inline validation at public export entry points."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from wenyi_core.assemble.inline_validation import InlinePreservationError
from wenyi_core.assemble.writer import assemble
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.storage.file import FileStorage


def _create_test_store_with_inline(
    tmpdir: Path, *, has_target_inline: bool, empty_ranges: bool = False, corrupted: bool = False
) -> FileStorage:
    """Create a test store with inline markup."""
    store = FileStorage(str(tmpdir / "book"))

    # Segment with source inline
    seg = Segment(
        index=0,
        source="This is an <i>example</i> text.",
        target="这是一个示例文本。" if has_target_inline or empty_ranges or corrupted else None,
        meta={
            "epub_inline": {
                "version": 1,
                "source_length": 29,
                "nodes": [{"tag": "i", "offset": 11, "placement": "inline"}],
            }
        },
    )

    if has_target_inline:
        # Valid target_inline
        seg.meta["target_inline"] = {
            "version": 1,
            "target_length": 10,
            "ranges": [{"start": 4, "end": 6, "tag": "i"}],
            "source_node_count": 1,
            "target_node_count": 1,
            "validation_warnings": [],
        }
    elif empty_ranges:
        # Preservation failed - empty ranges
        seg.meta["target_inline"] = {
            "version": 1,
            "target_length": 10,
            "ranges": [],
            "source_node_count": 1,
            "target_node_count": 0,
            "preservation_failed": True,
            "validation_warnings": ["All 1 inline markers lost during translation/polish"],
        }
    elif corrupted:
        # Decode failed
        seg.meta["target_inline"] = {
            "version": 1,
            "decode_failed": True,
            "preservation_failed": True,
            "error": "Unmatched [[I_END]]",
            "source_node_count": 1,
        }

    chapter = Chapter(index=0, segments=[seg])
    store.save_chapter(chapter)

    manifest = {
        "fmt": "text",
        "source_lang": "en",
        "target_lang": "zh",
        "chapter_count": 1,
        "source_sha256": "abc123",
        "chapters": [{"index": 0, "title": "", "status": "done"}],
        "chapters": [{"index": 0, "title": "", "status": "done"}],
    }
    store.save_manifest(manifest)

    return store


def test_export_succeeds_with_valid_inline():
    """Export succeeds when inline preservation is valid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = _create_test_store_with_inline(tmp, has_target_inline=True)

        # Should not raise
        out_path = str(tmp / "output.txt")
        result = assemble(store, "input.txt", out_path=out_path, out_format="txt")
        assert Path(result).exists()


def test_export_fails_with_empty_ranges():
    """Export fails when inline markers were lost (empty ranges)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = _create_test_store_with_inline(tmp, has_target_inline=False, empty_ranges=True)

        out_path = str(tmp / "output.txt")
        with pytest.raises(InlinePreservationError) as exc_info:
            assemble(store, "input.txt", out_path=out_path, out_format="txt")

        error_msg = str(exc_info.value)
        assert "Chapter 0, Segment 0" in error_msg
        assert "preservation failed" in error_msg
        assert "Retranslate this segment" in error_msg

        # File should not be created
        assert not Path(out_path).exists()


def test_export_fails_with_corrupted_markers():
    """Export fails when marker decode failed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = _create_test_store_with_inline(tmp, has_target_inline=False, corrupted=True)

        out_path = str(tmp / "output.txt")
        with pytest.raises(InlinePreservationError) as exc_info:
            assemble(store, "input.txt", out_path=out_path, out_format="txt")

        error_msg = str(exc_info.value)
        assert "Chapter 0, Segment 0" in error_msg
        assert "preservation failed" in error_msg
        assert "Unmatched [[I_END]]" in error_msg

        # File should not be created
        assert not Path(out_path).exists()


def test_export_fails_with_missing_target_inline():
    """Export fails when segment has source inline but no target_inline (old plain state)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = _create_test_store_with_inline(tmp, has_target_inline=False)

        # Add target but no target_inline
        chapter = store.load_chapter(0)
        chapter.segments[0].target = "这是一个示例文本。"
        store.save_chapter(chapter)

        out_path = str(tmp / "output.txt")
        with pytest.raises(InlinePreservationError) as exc_info:
            assemble(store, "input.txt", out_path=out_path, out_format="txt")

        error_msg = str(exc_info.value)
        assert "Chapter 0, Segment 0" in error_msg
        assert "no verified target positions" in error_msg
        assert "retranslate" in error_msg.lower()


def test_export_succeeds_with_no_source_inline():
    """Export succeeds for segments without source inline markup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = FileStorage(str(tmp / "book"))

        # Segment without inline
        seg = Segment(
            index=0,
            source="Normal text without emphasis.",
            target="普通文本无强调。",
            meta={},
        )

        chapter = Chapter(index=0, segments=[seg])
        store.save_chapter(chapter)

        manifest = {
            "fmt": "text",
            "source_lang": "en",
            "target_lang": "zh",
            "chapter_count": 1,
            "source_sha256": "abc123",
        "chapters": [{"index": 0, "title": "", "status": "done"}],
        }
        store.save_manifest(manifest)

        # Should succeed
        out_path = str(tmp / "output.txt")
        result = assemble(store, "input.txt", out_path=out_path, out_format="txt")
        assert Path(result).exists()


def test_export_multiple_formats_fail_closed():
    """Test fail-closed validation applies to all formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = _create_test_store_with_inline(tmp, has_target_inline=False, empty_ranges=True)

        formats = ["txt", "markdown"]
        for fmt in formats:
            out_path = str(tmp / f"output.{fmt}")
            with pytest.raises(InlinePreservationError) as exc_info:
                assemble(store, f"input.{fmt}", out_path=out_path, out_format=fmt)

            error_msg = str(exc_info.value)
            assert "preservation failed" in error_msg or "markers were lost" in error_msg
            assert not Path(out_path).exists(), f"Format {fmt} created file despite validation error"


def test_mixed_segments_identifies_failures():
    """Export identifies only segments with inline preservation failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = FileStorage(str(tmp / "book"))

        segments = [
            # Segment 0: No inline, should pass
            Segment(index=0, source="Normal text.", target="普通文本。", meta={}),
            # Segment 1: Has inline with valid target_inline, should pass
            Segment(
                index=1,
                source="Text with <i>emphasis</i>.",
                target="带强调的文本。",
                meta={
                    "epub_inline": {
                        "version": 1,
                        "source_length": 25,
                        "nodes": [{"tag": "i", "offset": 10, "placement": "inline"}],
                    },
                    "target_inline": {
                        "version": 1,
                        "target_length": 8,
                        "ranges": [{"start": 1, "end": 3, "tag": "i"}],
                        "source_node_count": 1,
                        "target_node_count": 1,
                        "validation_warnings": [],
                    },
                },
            ),
            # Segment 2: Has inline but empty ranges, should FAIL
            Segment(
                index=2,
                source="Another <strong>bold</strong> text.",
                target="另一个粗体文本。",
                meta={
                    "epub_inline": {
                        "version": 1,
                        "source_length": 35,
                        "nodes": [{"tag": "strong", "offset": 8, "placement": "inline"}],
                    },
                    "target_inline": {
                        "version": 1,
                        "target_length": 8,
                        "ranges": [],
                        "source_node_count": 1,
                        "target_node_count": 0,
                        "preservation_failed": True,
                        "validation_warnings": [],
                    },
                },
            ),
        ]

        chapter = Chapter(index=0, segments=segments)
        store.save_chapter(chapter)

        manifest = {
            "fmt": "text",
            "source_lang": "en",
            "target_lang": "zh",
            "chapter_count": 1,
            "source_sha256": "abc123",
        "chapters": [{"index": 0, "title": "", "status": "done"}],
        }
        store.save_manifest(manifest)

        out_path = str(tmp / "output.txt")
        with pytest.raises(InlinePreservationError) as exc_info:
            assemble(store, "input.txt", out_path=out_path, out_format="txt")

        error_msg = str(exc_info.value)
        # Should only complain about segment 2
        assert "Segment 2" in error_msg
        assert "Segment 0" not in error_msg
        assert "Segment 1" not in error_msg
        assert "preservation failed" in error_msg or "markers were lost" in error_msg
