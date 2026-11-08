"""Track inline metadata through real pipeline: ingest → translate → polish → state → export.

Minimal test to verify epub_inline metadata generates target_inline after translation.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

from wenyi_core.config import Config
from wenyi_core.ingest.epub_reader import read_epub
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.markup.contracts import INLINE_META_KEY
from wenyi_core.pipeline.orchestrator import Orchestrator

from tests.fake_llm import routing_handler


def _create_minimal_epub_with_emphasis(epub_path: str) -> None:
    """Create minimal valid EPUB with emphasis that must be annotated."""
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
  <h1>Chapter Title</h1>
  <p>First paragraph with <em>emphasis</em> word.</p>
  <p>Second paragraph with <strong>bold</strong> text.</p>
</body>
</html>""",
        )


def test_track_inline_metadata_through_pipeline():
    """Track inline metadata: ingest → annotation → translate → target_inline → export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        _create_minimal_epub_with_emphasis(epub_path)

        print("\n=== Step 1: Real EPUB Ingest ===")
        doc = read_epub(epub_path, source_lang="ja", target_lang="zh")

        print(f"Chapters: {len(doc.chapters)}")
        ch = doc.chapters[0]
        print(f"Chapter 0: {ch.title}, {len(ch.segments)} segments")

        # Check if segments have epub_inline metadata
        for seg in ch.segments:
            inline_meta = seg.meta.get(INLINE_META_KEY)
            print(f"  Segment {seg.index}: source={seg.source[:40]!r}")
            print(f"    has {INLINE_META_KEY}: {inline_meta is not None}")
            if inline_meta:
                print(f"    nodes: {inline_meta.get('nodes', [])}")

        # Verify at least one segment has inline markup
        has_inline = any(seg.meta.get(INLINE_META_KEY) for seg in ch.segments)
        assert has_inline, "Real ingest must generate epub_inline metadata for emphasis"

        print("\n=== Step 2: Run Translation Pipeline ===")
        state_dir = str(tmp / "state")
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"polish": True, "review": False, "review_autofix": False},
                "paths": {"state_dir": state_dir},
            }
        )

        # Handler that preserves inline markers
        def inline_preserving_handler(messages, tier, json_mode):
            import re

            system = messages[0]["content"]
            user = messages[-1]["content"]

            # For translation/polish, preserve inline markers
            if "translator" in system.lower() or "prose editor" in system.lower():
                # Extract paragraphs with markers preserved
                paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE | re.DOTALL)

                print(f"\n[HANDLER DEBUG] Found {len(paragraphs)} paragraphs")
                for idx, text in paragraphs[:2]:
                    print(f"  [{idx}] {text[:100]}")

                if paragraphs:
                    # Check if it's polish request
                    if "polish" in user.lower() or "Polish the translations" in user:
                        # Polish is a continue operation, get previous translations
                        for msg in reversed(messages[:-1]):
                            if msg.get("role") == "assistant":
                                try:
                                    prev = json.loads(msg["content"])
                                    trans = prev.get("translations", [])
                                    if trans:
                                        # Polish: preserve ALL markers and slightly modify text
                                        polished = []
                                        for t in trans:
                                            # Keep markers exactly, modify text between markers
                                            marker_pattern = r"(\[\[[A-Z_]+_(START|END)\]\])"
                                            parts = re.split(marker_pattern, t)
                                            result = []
                                            for part in parts:
                                                if re.match(marker_pattern, part):
                                                    # Keep marker unchanged
                                                    result.append(part)
                                                elif part.strip():
                                                    # Modify text (add "润" prefix)
                                                    result.append(f"润{part}")
                                                else:
                                                    result.append(part)
                                            polished.append("".join(result))

                                        print(f"[HANDLER DEBUG] Polish returning: {polished[:2]}")
                                        return json.dumps(
                                            {"polished": polished}, ensure_ascii=False
                                        )
                                except Exception as e:
                                    print(f"[HANDLER DEBUG] Polish error: {e}")
                                    pass

                    # Translation: extract and preserve markers
                    translations = []
                    for idx, text in paragraphs:
                        # Simple strategy: find pairs of START/END and keep them around translated text
                        # Example: "word [[EM_START]]emphasis[[EM_END]] word" -> "词[[EM_START]]强调[[EM_END]]词"

                        marker_pattern = r"\[\[([A-Z_]+)_(START|END)\]\]"
                        parts = re.split(marker_pattern, text)

                        result = []
                        i = 0
                        while i < len(parts):
                            part = parts[i]
                            if i + 2 < len(parts) and parts[i + 2] in ("START", "END"):
                                # This is text before a marker
                                if part.strip():
                                    result.append(f"译{idx}")
                                # Next parts are marker components: tag_name, START/END
                                tag = parts[i + 1]
                                direction = parts[i + 2]
                                result.append(f"[[{tag}_{direction}]]")
                                i += 3
                            else:
                                # Regular text
                                if part.strip():
                                    result.append(f"译{idx}")
                                i += 1

                        translations.append("".join(result))

                    print(f"[HANDLER DEBUG] Returning translations: {translations[:3]}")
                    return json.dumps({"translations": translations}, ensure_ascii=False)

            # Fallback to routing_handler for other operations
            return routing_handler(messages, tier, json_mode)

        client = FakeClient(handler=inline_preserving_handler)
        orch = Orchestrator(config, client=client)
        store = orch.run(epub_path)

        print("\n=== Step 3: Verify State After Translation ===")
        manifest = store.load_manifest()
        print(f"Manifest chapters: {len(manifest['chapters'])}")

        # Reload chapter from disk
        translated_ch = store.load_chapter(0)
        print(f"Translated chapter: {len(translated_ch.segments)} segments")

        for seg in translated_ch.segments:
            print(f"  Segment {seg.index}:")
            print(f"    source: {seg.source[:40]}")
            print(f"    target: {seg.target[:40] if seg.target else None}")

            epub_inline = seg.meta.get(INLINE_META_KEY)
            target_inline = seg.meta.get("target_inline")

            print(f"    has epub_inline: {epub_inline is not None}")
            print(f"    has target_inline: {target_inline is not None}")

            if target_inline:
                print(f"    target_inline keys: {list(target_inline.keys())}")
                print(f"    target_inline full: {target_inline}")
                items = target_inline.get("items") or target_inline.get("ranges") or []
                print(f"    inline items/ranges: {len(items)}")
                for item in items[:2]:
                    print(f"      - {item}")

        # Verify target_inline exists for segments with emphasis
        segments_with_inline = [
            seg for seg in translated_ch.segments if seg.meta.get(INLINE_META_KEY)
        ]
        segments_with_target_inline = [
            seg for seg in translated_ch.segments if seg.meta.get("target_inline")
        ]

        print(f"\nSegments with epub_inline: {len(segments_with_inline)}")
        print(f"Segments with target_inline: {len(segments_with_target_inline)}")

        # This is the KEY assertion: translated segments with emphasis must have target_inline
        for seg in segments_with_inline:
            if seg.target:  # Only check translated segments
                target_inline = seg.meta.get("target_inline")
                assert target_inline is not None, (
                    f"Segment {seg.index} has epub_inline but missing target_inline after translation. "
                    f"Source: {seg.source[:50]}, Target: {seg.target[:50] if seg.target else None}"
                )
                assert isinstance(target_inline, dict), "target_inline must be dict"
                assert "ranges" in target_inline, "target_inline must have ranges"

        print("\n=== Step 4: Verify Export ===")
        # Export Markdown
        md_path = str(tmp / "output.md")
        orch.run_assemble(epub_path, out_path=md_path, out_format="markdown")

        md_content = Path(md_path).read_text(encoding="utf-8")
        print(f"Markdown length: {len(md_content)}")
        print(f"Has emphasis markers: {'*' in md_content or '_' in md_content}")

        # Export EPUB
        epub_out = str(tmp / "output.epub")
        orch.run_assemble(epub_path, out_path=epub_out, out_format="epub")

        with zipfile.ZipFile(epub_out, "r") as zf:
            ch_html = zf.read("OEBPS/c1.xhtml").decode("utf-8")
            print(f"EPUB chapter HTML length: {len(ch_html)}")
            print(f"Has em tags: {'<em>' in ch_html}")
            print(f"Has strong tags: {'<strong>' in ch_html}")

        print("\n=== SUCCESS: inline metadata preserved through pipeline ===")


if __name__ == "__main__":
    test_track_inline_metadata_through_pipeline()
