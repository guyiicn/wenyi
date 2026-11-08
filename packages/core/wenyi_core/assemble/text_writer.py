"""Plain-text and Markdown output.
Read RunStore chapters, merge continuations with _merged_paragraphs and produce monolingual
or bilingual text according to bilingual and order.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import unquote

from ..models.document import KIND_HEADING
from .export_view import AssembleStore
from .writer_common import _bilingual_source, _merged_paragraphs, _ordered_pair


def _safe_filename(name: str, fallback: str = "file") -> str:
    """Sanitize filename for cross-platform safety."""
    # Remove path separators and control characters
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Trim and handle empty
    safe = safe.strip('. ')
    return safe if safe else fallback


def _extract_images_from_epub(
    source_path: str,
    assets_dir: str,
) -> dict[str, str]:
    """Extract images from EPUB to assets directory, return mapping of original path to MD path."""
    if not os.path.exists(source_path) or not source_path.lower().endswith('.epub'):
        return {}
    
    os.makedirs(assets_dir, exist_ok=True)
    image_map: dict[str, str] = {}
    used_names: set[str] = set()
    
    try:
        with zipfile.ZipFile(source_path, 'r') as zf:
            for info in zf.infolist():
                # Check for image extensions
                lower_name = info.filename.lower()
                if not any(lower_name.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp')):
                    continue
                
                # Get base name
                base_name = os.path.basename(info.filename)
                if not base_name:
                    continue
                
                # Handle duplicates by adding suffix
                name_without_ext, ext = os.path.splitext(base_name)
                safe_name = _safe_filename(name_without_ext, 'image') + ext.lower()
                final_name = safe_name
                counter = 1
                while final_name in used_names:
                    final_name = f"{_safe_filename(name_without_ext, 'image')}_{counter}{ext.lower()}"
                    counter += 1
                used_names.add(final_name)
                
                # Extract image
                target_path = os.path.join(assets_dir, final_name)
                with open(target_path, 'wb') as f:
                    f.write(zf.read(info.filename))
                
                # Map original EPUB path to MD relative path
                # Normalize EPUB path separators
                normalized_epub_path = info.filename.replace('\\', '/')
                image_map[normalized_epub_path] = f"assets/{final_name}"
    
    except Exception:
        # If extraction fails, return empty map
        return {}
    
    return image_map


def _assemble_plain_text(
    store: AssembleStore,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
    markdown: bool = False,
) -> str:
    """Shared text/Markdown implementation; prefix headings with # when markdown=True."""
    m = store.load_manifest()
    
    # Extract images for Markdown
    image_map: dict[str, str] = {}
    if markdown:
        source_path = m.get("source_path")
        if isinstance(source_path, str) and source_path:
            # Create assets directory next to output
            out_dir = os.path.dirname(os.path.abspath(out_path))
            assets_dir = os.path.join(out_dir, "assets")
            image_map = _extract_images_from_epub(source_path, assets_dir)
    
    chapter_blocks: list[str] = []
    for c in m["chapters"]:
        ch = store.load_chapter(c["index"])
        blocks: list[str] = []
        for kind, target, source, level in _merged_paragraphs(ch):
            if kind == KIND_HEADING and markdown:
                target = "#" * min(level, 6) + " " + target
            src = _bilingual_source(source, target) if (bilingual and kind != KIND_HEADING) else ""
            if not src:
                blocks.append(target)
            else:
                first, second = _ordered_pair(src, target, order)
                blocks.extend((first, second))
        
        # TODO: Add image references in markdown (requires segment-level image tracking)
        # For now, images are extracted but not referenced in the text
        
        chapter_blocks.append("\n\n".join(blocks))
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chapter_blocks) + "\n")
    return out_path


# Plain text.
def _assemble_text(
    store: AssembleStore,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """Rebuild UTF-8 text by chapter and paragraph, optionally including bilingual source text."""
    return _assemble_plain_text(store, out_path, bilingual=bilingual, order=order)


# ── markdown ──────────────────────────────────────────────────────────────────
def _assemble_markdown(
    store: AssembleStore,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """Rebuild Markdown with # headings and optional bilingual source text."""
    return _assemble_plain_text(store, out_path, bilingual=bilingual, order=order, markdown=True)
