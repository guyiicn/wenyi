"""Test real Chapter One image preservation through pipeline."""

import json
import re
import tempfile
from pathlib import Path

from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator


def _fake_ch1_handler(messages, tier, json_mode):
    """Handler matching fake_real.py logic."""
    user = messages[-1]["content"]
    
    # Check for numbered paragraphs (translation)
    ps = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.M)
    if ps:
        out = []
        for _, t in ps:
            parts = re.split(r"(\[\[[A-Z_0-9]+_(?:START|END)\]\])", t)
            out.append("".join(
                p if p.startswith("[[") else ("伪" * max(1, len(p) // 4) if p.strip() else p)
                for p in parts
            ))
        
        # Check if this is title translation
        allm = " ".join(m["content"] for m in messages)
        if "TOC title" in allm or "chapter title" in allm.lower():
            return json.dumps(["第一章(伪)"] * len(out), ensure_ascii=False)
        
        return json.dumps({"translations": out}, ensure_ascii=False)
    
    # Check for title prompts
    if "itle" in user:
        return json.dumps(["第一章(伪)"], ensure_ascii=False)
    
    # Fallback
    return json.dumps({"translations": ["伪"], "titles": ["伪章"]}, ensure_ascii=False)


def test_real_ch1_image_in_markdown():
    """Test real Chapter One EPUB: image appears once in correct position in MD output."""
    real_epub = Path("/home/ubuntu/.cursor/projects/workspace/uploads/Light_of_the_Jedi.Chapter_One.en.epub")
    
    if not real_epub.exists():
        import pytest
        pytest.skip("Real Chapter One EPUB not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        state_dir = str(tmp / "state")
        
        config = Config.from_dict({
            "language": {"source": "en", "target": "zh"},
            "llm": {"preset": "fake"},
            "pipeline": {
                "polish": False,
                "review": False,
                "review_autofix": False,
                "book_understanding": False,
            },
            "paths": {"state_dir": state_dir},
        })
        
        client = FakeClient(handler=_fake_ch1_handler)
        orch = Orchestrator(config, client=client)
        
        # Run translation
        store = orch.run(str(real_epub))
        
        # Verify chapter structure
        manifest = store.load_manifest()
        assert manifest["fmt"] == "epub"
        assert len(manifest["chapters"]) == 1
        
        chapter = store.load_chapter(0)
        
        # Verify 71 segments
        text_segs = [s for s in chapter.segments if s.kind == "text"]
        assert len(text_segs) == 71, f"Expected 71 text segments, got {len(text_segs)}"
        
        # Check image_blocks in metadata
        image_blocks = chapter.meta.get("image_blocks")
        print(f"\n=== Chapter metadata image_blocks ===")
        print(f"image_blocks: {image_blocks}")
        
        # Check first segment's resource_href for normalization
        if chapter.segments:
            first_seg = chapter.segments[0]
            print(f"First segment resource_href: {first_seg.resource_href}")
        
        if image_blocks:
            # Should NOT have duplicates at position 0 and 1
            positions = [block.get("position") for block in image_blocks if isinstance(block, dict)]
            print(f"Image positions: {positions}")
            
            # Debug: compute what normalized paths would be
            if chapter.segments:
                first_href = chapter.segments[0].resource_href
                for block in image_blocks:
                    if isinstance(block, dict):
                        images = block.get("images", [])
                        for img in images:
                            src = img.get("src", "")
                            if src and first_href:
                                # Simulate normalization
                                import os
                                from urllib.parse import urljoin
                                base_dir = os.path.dirname(first_href)
                                joined = urljoin(base_dir + "/", src)
                                normalized = joined.replace("\\", "/").lstrip("./")
                                print(f"  Image src={src} -> normalized={normalized}")
            
            # Each image should appear exactly once
            from collections import Counter
            pos_counts = Counter(positions)
            for pos, count in pos_counts.items():
                assert count == 1, f"Image at position {pos} appears {count} times (duplicate)"
        
        # Export Markdown
        md_path = str(tmp / "output.md")
        orch.run_assemble(str(real_epub), out_path=md_path, out_format="markdown")
        
        md_content = Path(md_path).read_text(encoding="utf-8")
        print(f"\n=== Markdown content (first 500 chars) ===")
        print(md_content[:500])
        
        # Count image references in Markdown
        # Pattern: ![...](assets/...)
        image_refs = re.findall(r'!\[[^\]]*\]\(assets/[^)]+\)', md_content)
        print(f"\n=== Image references in MD ===")
        print(f"Count: {len(image_refs)}")
        for ref in image_refs:
            print(f"  {ref}")
        
        # STRICT: Must have exactly 1 image reference (chapter title image)
        assert len(image_refs) == 1, \
            f"Expected 1 image reference, got {len(image_refs)}"
        
        # Verify assets directory exists
        assets_dir = tmp / "assets"
        assert assets_dir.exists(), "assets/ directory not created"
        
        # Verify actual image file exists
        images = list(assets_dir.glob("*.jpg")) + list(assets_dir.glob("*.png"))
        assert len(images) >= 1, f"No image files in assets/, found: {list(assets_dir.iterdir())}"
        
        # Verify image file hash matches original
        import hashlib
        original_jpg = Path("/home/ubuntu/.cursor/projects/workspace/uploads")
        # Extract from EPUB to compare
        import zipfile
        with zipfile.ZipFile(real_epub, "r") as zf:
            jpg_files = [n for n in zf.namelist() if n.endswith(".jpg")]
            assert len(jpg_files) >= 1, "No jpg in original EPUB"
            
            original_data = zf.read(jpg_files[0])
            original_hash = hashlib.sha256(original_data).hexdigest()
            
            # Check exported image hash
            exported_img = images[0]
            exported_data = exported_img.read_bytes()
            exported_hash = hashlib.sha256(exported_data).hexdigest()
            
            assert exported_hash == original_hash, \
                f"Image hash mismatch: original={original_hash[:16]}... exported={exported_hash[:16]}..."
        
        # Verify MD title is translated (from fake handler)
        # Should have "第一章(伪)" not "Chapter One"
        lines = md_content.split("\n")
        title_lines = [l for l in lines if l.startswith("#")]
        
        if title_lines:
            first_title = title_lines[0]
            print(f"\n=== First title line ===\n{first_title}")
            
            # Should contain translated title
            assert "第一章" in first_title or "伪" in first_title, \
                f"Title not translated: {first_title}"
            
            # Should NOT have duplicate English title after translated title
            # Check next non-empty line
            title_idx = lines.index(first_title)
            next_lines = [l.strip() for l in lines[title_idx+1:title_idx+5] if l.strip()]
            
            # Should not have "Chapter One" as a separate line after the title
            for line in next_lines[:2]:  # Check first 2 non-empty lines after title
                if line.startswith("#"):
                    continue  # Another heading is ok
                assert "Chapter One" not in line, \
                    f"Duplicate English title after translated title: {line}"


def test_image_reference_position_matches_source():
    """Verify image appears at correct position matching source structure."""
    real_epub = Path("/home/ubuntu/.cursor/projects/workspace/uploads/Light_of_the_Jedi.Chapter_One.en.epub")
    
    if not real_epub.exists():
        import pytest
        pytest.skip("Real Chapter One EPUB not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        state_dir = str(tmp / "state")
        
        config = Config.from_dict({
            "language": {"source": "en", "target": "zh"},
            "llm": {"preset": "fake"},
            "pipeline": {"polish": False, "review": False, "review_autofix": False},
            "paths": {"state_dir": state_dir},
        })
        
        client = FakeClient(handler=_fake_ch1_handler)
        orch = Orchestrator(config, client=client)
        
        store = orch.run(str(real_epub))
        chapter = store.load_chapter(0)
        
        # Find image block position
        image_blocks = chapter.meta.get("image_blocks", [])
        if not image_blocks:
            import pytest
            pytest.fail("No image_blocks in chapter metadata")
        
        # Should be at position 0 (chapter title image)
        first_image = image_blocks[0]
        assert first_image.get("position") == 0, \
            f"First image should be at position 0, got {first_image.get('position')}"
        
        # Export and verify image appears at top (near title)
        md_path = str(tmp / "output.md")
        orch.run_assemble(str(real_epub), out_path=md_path, out_format="markdown")
        
        md_content = Path(md_path).read_text(encoding="utf-8")
        lines = md_content.split("\n")
        
        # Find first image reference line
        img_line_idx = None
        for idx, line in enumerate(lines):
            if line.startswith("![") and "assets/" in line:
                img_line_idx = idx
                break
        
        assert img_line_idx is not None, "No image reference found in MD"
        
        # Image should be near the top (within first 10 lines after title)
        assert img_line_idx < 10, \
            f"Image appears too late at line {img_line_idx}, should be near top"
