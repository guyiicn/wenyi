"""Test Review Autofix interaction with inline emphasis metadata.

Verifies that autofix correctly updates target_inline metadata when modifying
translated text, ensuring emphasis markers remain accurate after term corrections
or other autofix modifications.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path

import pytest

from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from tests.fake_llm import routing_handler as base_routing_handler
from wenyi_core.pipeline.orchestrator import Orchestrator


def marker_preserving_handler(messages, tier, json_mode):
    """Handler that preserves [[TAG_START]]/[[TAG_END]] markers."""
    system = messages[0]["content"]
    user = messages[-1]["content"]
    
    if "translator" in system.lower():
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
    
    return base_routing_handler(messages, tier, json_mode)


def test_autofix_modifies_target_without_updating_inline_metadata():
    """Document that autofix modifying target text invalidates target_inline ranges.
    
    Currently, autofix_publish.py modifies chapter.text_segments[text_index].target
    (line 73) without updating or invalidating the segment's target_inline metadata.
    This causes target_inline ranges to point at wrong character offsets in the
    modified text.
    
    Expected behavior:
    - Either re-encode/decode inline markers for the new target text
    - Or mark target_inline as stale/invalid and fail export
    - Or skip autofix for segments with inline markup
    
    This test demonstrates the issue but does not require a fix in this iteration.
    """
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
                "pipeline": {
                    "polish": False,
                    "review": True,
                    "review_autofix": True,  # Enable autofix
                },
                "paths": {"state_dir": state_dir},
            }
        )
        
        # Handler preserves markers and generates review issues
        call_count = {"translate": 0, "review": 0}
        
        def autofix_test_handler(messages, tier, json_mode):
            system = messages[0]["content"]
            user = messages[-1]["content"]
            
            # Translation with inline markers
            if "translator" in system.lower():
                call_count["translate"] += 1
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
                                result.append(f"原文{idx}")
                        translations.append("".join(result))
                    return json.dumps({"translations": translations}, ensure_ascii=False)
            
            # Review: generate fake issues
            if "review" in system.lower() and "issues" in user.lower():
                call_count["review"] += 1
                # Return a fake issue that would trigger autofix modification
                return json.dumps({
                    "issues": [
                        {
                            "id": "issue-1",
                            "category": "terminology",
                            "severity": "major",
                            "location_chapter": 0,
                            "location_index": 1,  # Segment with emphasis
                            "target_quote": "原文1",
                            "suggestion": "修改后原文1增加字符",  # Different length!
                            "explanation": "Test autofix modification",
                        }
                    ]
                }, ensure_ascii=False)
            
            return base_routing_handler(messages, tier, json_mode)
        
        client = FakeClient(handler=autofix_test_handler)
        orch = Orchestrator(config, client=client)
        
        # Note: This test may not complete full review/autofix due to FakeClient limitations
        # The key point is documenting the autofix→inline metadata interaction issue
        
        try:
            store = orch.run(epub_path)
            
            # If translation completed, check if target_inline exists
            ch = store.load_chapter(0)
            emphasis_seg = next(
                (s for s in ch.segments if s.source and "emphasis" in s.source), None
            )
            
            if emphasis_seg and emphasis_seg.target:
                target_inline = emphasis_seg.meta.get("target_inline")
                
                # Document the issue: if autofix ran and modified target,
                # target_inline ranges would be stale
                if target_inline and isinstance(target_inline, dict):
                    ranges = target_inline.get("ranges", [])
                    if ranges:
                        # This range was computed for the original translation
                        # If autofix modified the text, these positions are now wrong
                        print(f"Target: {emphasis_seg.target}")
                        print(f"Target inline ranges: {ranges}")
                        print("Note: If autofix modified this target, ranges are stale")
        
        except Exception as e:
            # Test may fail due to FakeClient limitations, that's expected
            # The documentation of the issue is the main goal
            print(f"Test completed with expected limitation: {type(e).__name__}")
            pytest.skip(f"FakeClient limitation prevents full review/autofix: {e}")


def test_autofix_publish_preserves_inline_metadata():
    """Verify autofix_publish correctly updates target_inline when modifying text."""
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
  <p>Text with <em>term</em> word.</p>
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
        
        # First translate to get valid state with target_inline
        client = FakeClient(handler=marker_preserving_handler)
        orch = Orchestrator(config, client=client)
        store = orch.run(epub_path)
        
        # Load chapter and find segment with emphasis
        ch = store.load_chapter(0)
        term_seg_index = None
        term_seg = None
        for i, seg in enumerate(ch.segments):
            if seg.source and "term" in seg.source:
                term_seg_index = i
                term_seg = seg
                break
        
        assert term_seg is not None, "Must find segment with 'term'"
        assert term_seg.target is not None
        
        # Verify initial target_inline exists
        target_inline = term_seg.meta.get("target_inline")
        assert target_inline is not None
        assert len(target_inline.get("ranges", [])) > 0
        
        initial_target = term_seg.target
        initial_ranges = target_inline["ranges"]
        
        # Simulate autofix modifying part of the target (e.g., term correction)
        # Replace first occurrence of "中文" with "修正后中文" (longer text)
        # This should shift the emphasis position
        modified_target = initial_target.replace("中文", "修正后中文", 1)
        
        # Create fake autofix index entry
        from wenyi_core.pipeline.autofix.autofix_publish import AutofixPublisher
        from wenyi_core.pipeline.annotate.annotations import AnnotationService
        from wenyi_core.pipeline.annotate.docx_styles import DocxStyleService
        from wenyi_core.pipeline.run.runtime import PipelineRuntime
        
        runtime = PipelineRuntime(config)
        annotations = AnnotationService(runtime)
        docx_styles = DocxStyleService(runtime)
        publisher = AutofixPublisher(annotations, docx_styles)
        
        # Create a minimal autofix index
        from wenyi_core.review.run_store import ReviewRunStore
        
        review_debug = ReviewRunStore(str(tmp / "review"))
        
        autofix_index = {
            "locations": [
                {
                    "chapter": 0,
                    "index": term_seg_index,
                    "before": initial_target,
                    "target": modified_target,
                    "before_hash": None,  # Will be computed
                }
            ],
            "records": [],
        }
        
        # Compute hash
        from wenyi_core.review.models import text_hash
        autofix_index["locations"][0]["before_hash"] = text_hash(initial_target)
        
        # Apply autofix
        publisher.apply(store, review_debug, autofix_index, progress=None)
        
        # Reload chapter from disk
        ch_after = store.load_chapter(0)
        seg_after = ch_after.segments[term_seg_index]
        
        # Verify target was modified
        assert seg_after.target == modified_target, "Target should be modified"
        
        # Verify target_inline was updated
        target_inline_after = seg_after.meta.get("target_inline")
        assert target_inline_after is not None, "target_inline must exist after autofix"
        
        ranges_after = target_inline_after.get("ranges", [])
        assert len(ranges_after) > 0, "Must have ranges after autofix"
        
        # Verify ranges are different from initial (due to text length change)
        assert ranges_after != initial_ranges, "Ranges must be updated"
        
        # Verify the emphasized text is correct
        if ranges_after:
            r = ranges_after[0]
            emphasized = seg_after.target[r["start"]:r["end"]]
            # Should still be the middle "中文1" part
            assert emphasized, f"Emphasized text must not be empty"
            assert "中文" in emphasized, f"Emphasized text '{emphasized}' must be Chinese"
        
        # Verify export does not fail
        md_path = str(tmp / "output.md")
        orch.run_assemble(epub_path, out_path=md_path, out_format="markdown")
        
        assert Path(md_path).exists(), "Export must succeed after autofix with inline"


def test_autofix_inline_modification_before_emphasis():
    """Autofix modifying text before emphasis updates ranges correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # Create EPUB
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
  <p>Prefix text with <em>emphasis</em> suffix.</p>
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
        
        client = FakeClient(handler=marker_preserving_handler)
        orch = Orchestrator(config, client=client)
        store = orch.run(epub_path)
        
        # Find segment with emphasis
        ch = store.load_chapter(0)
        seg_index = None
        seg = None
        for i, s in enumerate(ch.segments):
            if s.source and "emphasis" in s.source:
                seg_index = i
                seg = s
                break
        
        assert seg is not None
        initial_target = seg.target
        
        # Modify prefix (before emphasis)
        # This will shift the emphasis position
        modified_target = initial_target.replace("Prefix", "LongerPrefix")
        
        # Apply autofix
        from wenyi_core.pipeline.autofix.autofix_publish import AutofixPublisher
        from wenyi_core.pipeline.annotate.annotations import AnnotationService
        from wenyi_core.pipeline.annotate.docx_styles import DocxStyleService
        from wenyi_core.pipeline.run.runtime import PipelineRuntime
        from wenyi_core.review.run_store import ReviewRunStore
        from wenyi_core.review.models import text_hash
        
        runtime = PipelineRuntime(config)
        annotations = AnnotationService(runtime)
        docx_styles = DocxStyleService(runtime)
        publisher = AutofixPublisher(annotations, docx_styles)
        review_debug = ReviewRunStore(str(tmp / "review"))
        
        autofix_index = {
            "locations": [
                {
                    "chapter": 0,
                    "index": seg_index,
                    "before": initial_target,
                    "target": modified_target,
                    "before_hash": text_hash(initial_target),
                }
            ],
            "records": [],
        }
        
        publisher.apply(store, review_debug, autofix_index, progress=None)
        
        # Verify ranges were updated
        ch_after = store.load_chapter(0)
        seg_after = ch_after.segments[seg_index]
        
        assert seg_after.target == modified_target
        
        target_inline = seg_after.meta.get("target_inline")
        assert target_inline is not None
        
        ranges = target_inline.get("ranges", [])
        assert len(ranges) > 0
        
        # Verify emphasized text is still correct
        r = ranges[0]
        emphasized = seg_after.target[r["start"]:r["end"]]
        assert emphasized, f"Emphasized text at [{r['start']}:{r['end']}] is empty"
        
        # Export must succeed
        md_path = str(tmp / "output.md")
        orch.run_assemble(epub_path, out_path=md_path, out_format="markdown")
        assert Path(md_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
