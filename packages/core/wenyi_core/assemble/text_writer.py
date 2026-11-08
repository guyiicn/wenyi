"""Plain-text and Markdown output.
Read RunStore chapters, merge continuations with _merged_paragraphs and produce monolingual
or bilingual text according to bilingual and order.
"""

from __future__ import annotations

import os
import re
import zipfile
from urllib.parse import urljoin

from bs4 import BeautifulSoup, UnicodeDammit

from ..markup.contracts import INLINE_ID_ATTR, INLINE_META_KEY
from ..models.document import Segment
from .export_view import AssembleStore
from .markdown_inline import insert_markdown_emphasis
from .writer_common import _bilingual_source, _ordered_pair


def _safe_filename(name: str, fallback: str = "file") -> str:
    """Sanitize filename for cross-platform safety."""
    # Remove path separators and control characters
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    # Trim and handle empty
    safe = safe.strip(". ")
    return safe if safe else fallback


def _normalize_epub_href(base_href: str, img_src: str) -> str:
    """Resolve relative image href against EPUB resource base."""
    if not img_src:
        return ""
    # Handle absolute paths
    if img_src.startswith("/"):
        return img_src.lstrip("/")
    # Resolve relative to resource directory
    base_dir = os.path.dirname(base_href)
    joined = urljoin(base_dir + "/", img_src)
    # Remove leading './' and normalize
    normalized = joined.replace("\\", "/").lstrip("./")
    return normalized


def _extract_segment_images_from_epub(
    source_path: str,
    chapters: list,
    meta: dict[str, object],
) -> dict[tuple[int, int], list[tuple[str, str, str, str]]]:
    """Extract segment-level image info: (ch_idx, seg_idx) -> [(inline_id, src, alt, placement)]."""
    if not os.path.exists(source_path) or not source_path.lower().endswith(".epub"):
        return {}

    segment_images: dict[tuple[int, int], list[tuple[str, str, str, str]]] = {}

    try:
        with zipfile.ZipFile(source_path, "r") as zf:
            # Get resource specs
            raw_resources = meta.get("epub_resources")
            if not isinstance(raw_resources, list):
                return {}

            resource_hrefs: dict[str, str] = {}
            for res in raw_resources:
                if not isinstance(res, dict):
                    continue
                href = res.get("href")
                if isinstance(href, str) and href:
                    resource_hrefs[href] = href

            # Build segment -> resource mapping
            segment_to_resource: dict[tuple[int, int], str] = {}
            for ch in chapters:
                for seg in ch.segments:
                    if isinstance(seg, Segment) and seg.resource_href:
                        segment_to_resource[(ch.index, seg.index)] = seg.resource_href

            # Parse each resource and extract image metadata
            for href in resource_hrefs.keys():
                if href not in zf.namelist():
                    continue

                try:
                    source_data = zf.read(href)
                    html = UnicodeDammit(source_data).unicode_markup
                    if html is None:
                        html = source_data.decode("utf-8", errors="replace")

                    soup = BeautifulSoup(html, "html.parser")

                    # Find all img tags with inline IDs
                    for img in soup.find_all("img"):
                        inline_id = img.get(INLINE_ID_ATTR)
                        if not isinstance(inline_id, str):
                            continue

                        src = img.get("src", "")
                        alt = img.get("alt", "")

                        if not src:
                            continue

                        # Resolve relative src
                        normalized_src = _normalize_epub_href(href, str(src))

                        # Find which segment owns this inline_id
                        for (ch_idx, seg_idx), res_href in segment_to_resource.items():
                            if res_href != href:
                                continue

                            # Check if segment has this inline_id in its metadata
                            ch = next((c for c in chapters if c.index == ch_idx), None)
                            if not ch:
                                continue

                            seg = next((s for s in ch.segments if s.index == seg_idx), None)
                            if not seg or not isinstance(seg, Segment):
                                continue

                            raw_inline = seg.meta.get(INLINE_META_KEY)
                            if not isinstance(raw_inline, dict):
                                continue

                            raw_nodes = raw_inline.get("nodes")
                            if not isinstance(raw_nodes, list):
                                continue

                            # Check if this inline_id belongs to this segment and get placement
                            for node in raw_nodes:
                                if not isinstance(node, dict):
                                    continue
                                if node.get("id") == inline_id:
                                    placement = node.get("placement", "block")
                                    key = (ch_idx, seg_idx)
                                    if key not in segment_images:
                                        segment_images[key] = []
                                    segment_images[key].append(
                                        (inline_id, normalized_src, str(alt), placement)
                                    )
                                    break

                except Exception:
                    # Skip failed resources
                    continue

    except Exception:
        # If EPUB read fails, return empty
        return {}

    return segment_images


def _extract_images_from_epub(
    source_path: str,
    assets_dir: str,
) -> dict[str, str]:
    """Extract images from EPUB to assets directory, return mapping of normalized EPUB path to MD path."""
    if not os.path.exists(source_path) or not source_path.lower().endswith(".epub"):
        return {}

    os.makedirs(assets_dir, exist_ok=True)
    image_map: dict[str, str] = {}
    used_names: set[str] = set()

    try:
        with zipfile.ZipFile(source_path, "r") as zf:
            for info in zf.infolist():
                # Check for image extensions
                lower_name = info.filename.lower()
                if not any(
                    lower_name.endswith(ext)
                    for ext in (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp")
                ):
                    continue

                # Get base name
                base_name = os.path.basename(info.filename)
                if not base_name:
                    continue

                # Handle duplicates by adding suffix
                name_without_ext, ext = os.path.splitext(base_name)
                safe_name = _safe_filename(name_without_ext, "image") + ext.lower()
                final_name = safe_name
                counter = 1
                while final_name in used_names:
                    final_name = (
                        f"{_safe_filename(name_without_ext, 'image')}_{counter}{ext.lower()}"
                    )
                    counter += 1
                used_names.add(final_name)

                # Extract image
                target_path = os.path.join(assets_dir, final_name)
                with open(target_path, "wb") as f:
                    f.write(zf.read(info.filename))

                # Map normalized EPUB path to MD relative path
                normalized_epub_path = info.filename.replace("\\", "/")
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
    raw_meta = m.get("meta")
    meta = raw_meta if isinstance(raw_meta, dict) else {}

    # Extract images for Markdown
    image_map: dict[str, str] = {}
    segment_images: dict[tuple[int, int], list[tuple[str, str, str]]] = {}

    if markdown:
        source_path = m.get("source_path")
        if isinstance(source_path, str) and source_path:
            # Create assets directory next to output
            out_dir = os.path.dirname(os.path.abspath(out_path))
            assets_dir = os.path.join(out_dir, "assets")

            # Load chapters for image extraction
            chapters = [store.load_chapter(c["index"]) for c in m["chapters"]]

            # Extract image files and metadata
            image_map = _extract_images_from_epub(source_path, assets_dir)
            segment_images = _extract_segment_images_from_epub(source_path, chapters, meta)

    chapter_blocks: list[str] = []
    for c in m["chapters"]:
        ch = store.load_chapter(c["index"])
        blocks: list[str] = []

        # Add chapter title page if markdown and chapter has title
        if markdown and ch.title:
            blocks.append(f"# {ch.title}")

        # Process segments in order to maintain image/paragraph sequence
        for seg in ch.segments:
            if not seg.target:
                continue

            # Get images for this segment
            seg_images = segment_images.get((ch.index, seg.index), [])

            # Separate block-level and inline images
            block_images = []
            inline_images = []
            for inline_id, norm_src, alt, placement in seg_images:
                if placement == "inline":
                    inline_images.append((inline_id, norm_src, alt, placement))
                else:
                    block_images.append((inline_id, norm_src, alt, placement))

            # Fail if inline images are present (not supported in Markdown)
            if inline_images and markdown:
                raise ValueError(
                    f"Chapter {ch.index}, Segment {seg.index}: Inline images within text "
                    f"are not supported in Markdown export. Found {len(inline_images)} inline "
                    f"image(s). Please use EPUB format or remove inline images."
                )

            # Add block-level images before segment text
            for inline_id, norm_src, alt, placement in block_images:
                md_path = image_map.get(norm_src, "")
                if md_path:
                    alt_text = alt if alt else "Image"
                    blocks.append(f"![{alt_text}]({md_path})")

            # Apply markdown emphasis if available
            target_text = seg.target
            if markdown and not seg.cont:
                target_inline = seg.meta.get("target_inline")
                if target_inline:
                    target_text = insert_markdown_emphasis(target_text, target_inline)

            # Add segment text
            if bilingual and seg.source:
                src = _bilingual_source(seg.source, target_text)
                first, second = _ordered_pair(src, target_text, order)
                blocks.extend([first, second])
            else:
                blocks.append(target_text)

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
