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
    """Extract segment-level image info from metadata: (ch_idx, seg_idx) -> [(inline_id, src, alt, placement)]."""
    if not os.path.exists(source_path) or not source_path.lower().endswith(".epub"):
        return {}

    segment_images: dict[tuple[int, int], list[tuple[str, str, str, str]]] = {}

    try:
        with zipfile.ZipFile(source_path, "r") as zf:
            for ch in chapters:
                for seg in ch.segments:
                    if not isinstance(seg, Segment) or not seg.resource_href:
                        continue

                    href = seg.resource_href
                    if href not in zf.namelist():
                        continue

                    try:
                        source_data = zf.read(href)
                        html = UnicodeDammit(source_data).unicode_markup
                        if html is None:
                            html = source_data.decode("utf-8", errors="replace")

                        soup = BeautifulSoup(html, "html.parser")

                        # Get inline metadata for this segment
                        raw_inline = seg.meta.get(INLINE_META_KEY)
                        if not isinstance(raw_inline, dict):
                            continue

                        raw_nodes = raw_inline.get("nodes", [])
                        if not isinstance(raw_nodes, list):
                            continue

                        # Collect all img nodes from metadata
                        img_nodes = []
                        for node in raw_nodes:
                            if isinstance(node, dict) and node.get("tag") == "img":
                                img_nodes.append(node)

                        if not img_nodes:
                            continue

                        # Find all img tags in HTML (in document order)
                        all_imgs = soup.find_all("img")

                        # Match by trying annotated id first, then fall back to positional matching
                        key = (ch.index, seg.index)
                        for img_node in img_nodes:
                            inline_id = img_node.get("id")
                            if not isinstance(inline_id, str):
                                continue
                            placement = img_node.get("placement", "after")

                            # Try to find by annotated id attribute
                            img = soup.find("img", attrs={INLINE_ID_ATTR: inline_id})

                            # If not found (original EPUB without annotation), use positional matching
                            # by extracting all imgs and matching count
                            if not img and all_imgs:
                                # Count how many img nodes we've seen so far in metadata
                                img_index = sum(
                                    1
                                    for n in raw_nodes[: raw_nodes.index(img_node)]
                                    if isinstance(n, dict) and n.get("tag") == "img"
                                )
                                if img_index < len(all_imgs):
                                    img = all_imgs[img_index]

                            if img:
                                src = img.get("src", "")
                                alt = img.get("alt", "")
                                if src:
                                    normalized_src = _normalize_epub_href(href, str(src))
                                    if key not in segment_images:
                                        segment_images[key] = []
                                    segment_images[key].append(
                                        (inline_id, normalized_src, str(alt), placement)
                                    )
                    except Exception:
                        continue

    except Exception:
        return {}

    return segment_images


def _extract_images_from_epub(
    source_path: str,
    assets_dir: str,
    required_images: set[str],
) -> dict[str, str]:
    """Extract images from EPUB to assets directory, fail if required images are missing.

    Args:
        source_path: Path to EPUB file
        assets_dir: Directory to extract images to
        required_images: Set of normalized EPUB paths that must exist

    Returns:
        Mapping of normalized EPUB path to MD relative path

    Raises:
        FileNotFoundError: If any required image is missing from EPUB
    """
    if not os.path.exists(source_path) or not source_path.lower().endswith(".epub"):
        return {}

    os.makedirs(assets_dir, exist_ok=True)
    image_map: dict[str, str] = {}
    used_names: set[str] = set()
    found_images: set[str] = set()

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
                found_images.add(normalized_epub_path)

        # Check for missing required images
        missing_images = required_images - found_images
        if missing_images:
            missing_list = ", ".join(sorted(missing_images))
            raise FileNotFoundError(
                f"Required image files missing from EPUB: {missing_list}. "
                f"Cannot create Markdown output with broken image references."
            )

    except FileNotFoundError:
        raise
    except Exception as e:
        raise OSError(f"Failed to extract images from EPUB: {e}")

    return image_map


def _assemble_plain_text(
    store: AssembleStore,
    source_path: str,
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
        if isinstance(source_path, str) and source_path:
            # Create assets directory next to output
            out_dir = os.path.dirname(os.path.abspath(out_path))
            assets_dir = os.path.join(out_dir, "assets")

            # Load chapters for image extraction
            chapters = [store.load_chapter(c["index"]) for c in m["chapters"]]

            # Extract segment images metadata first
            segment_images = _extract_segment_images_from_epub(source_path, chapters, meta)

            # Collect required image paths from segment images and standalone image blocks
            required_images: set[str] = set()
            for images_list in segment_images.values():
                for _inline_id, norm_src, _alt, _placement in images_list:
                    required_images.add(norm_src)

            # Also collect from standalone image blocks in chapter metadata
            for ch in chapters:
                raw_image_blocks = ch.meta.get("image_blocks")
                if isinstance(raw_image_blocks, list):
                    for block in raw_image_blocks:
                        if isinstance(block, dict):
                            images = block.get("images")
                            if isinstance(images, list):
                                for img_info in images:
                                    if isinstance(img_info, dict):
                                        src = img_info.get("src", "")
                                        if src and ch.segments:
                                            # Use first segment's resource_href for normalization
                                            first_href = ch.segments[0].resource_href
                                            if first_href:
                                                normalized_src = _normalize_epub_href(
                                                    first_href, src
                                                )
                                                required_images.add(normalized_src)

            # Scan all HTML resources for any referenced images (fail-closed validation)
            epub_resources = meta.get("epub_resources")
            if isinstance(epub_resources, list):
                try:
                    with zipfile.ZipFile(source_path, "r") as zf:
                        for resource in epub_resources:
                            if not isinstance(resource, dict):
                                continue
                            href = resource.get("href")
                            if not href:
                                continue
                            media_type = resource.get("media_type", "")
                            # Check media type if present, otherwise assume HTML/XHTML based on extension
                            is_html = False
                            if media_type:
                                is_html = (
                                    "xhtml" in media_type.lower() or "html" in media_type.lower()
                                )
                            else:
                                lower_href = href.lower()
                                is_html = lower_href.endswith(".xhtml") or lower_href.endswith(
                                    ".html"
                                )
                            if not is_html:
                                continue
                            if href not in zf.namelist():
                                continue
                            try:
                                source_data = zf.read(href)
                                html = UnicodeDammit(source_data).unicode_markup
                                if html is None:
                                    html = source_data.decode("utf-8", errors="replace")
                                soup = BeautifulSoup(html, "html.parser")
                                for img in soup.find_all("img"):
                                    src = img.get("src", "")
                                    if src:
                                        normalized_src = _normalize_epub_href(href, str(src))
                                        required_images.add(normalized_src)
                            except Exception:
                                continue
                except Exception:
                    pass

            # Extract image files with validation
            image_map = _extract_images_from_epub(source_path, assets_dir, required_images)

    chapter_blocks: list[str] = []
    for c in m["chapters"]:
        ch = store.load_chapter(c["index"])
        blocks: list[str] = []

        # Add chapter title page if markdown and chapter has title
        if markdown:
            # Prefer translated title from manifest over chapter.title
            ch_info = c  # Current chapter dict from manifest
            title_translated = ch_info.get("title_translated", "")
            title = title_translated or ch.title
            if title:
                blocks.append(f"# {title}")

        # Get image blocks for this chapter
        raw_image_blocks = ch.meta.get("image_blocks")
        image_blocks_by_position: dict[int, list[dict[str, str]]] = {}
        if isinstance(raw_image_blocks, list):
            for block in raw_image_blocks:
                if isinstance(block, dict):
                    position = block.get("position")
                    images = block.get("images")
                    if isinstance(position, int) and isinstance(images, list):
                        image_blocks_by_position[position] = images

        # Process segments in order to maintain image/paragraph sequence
        position = 0
        for seg in ch.segments:
            # Output standalone image blocks at this position
            if markdown and position in image_blocks_by_position:
                for img_info in image_blocks_by_position[position]:
                    src = img_info.get("src", "")
                    alt = img_info.get("alt", "Image")
                    if src and seg.resource_href:
                        normalized_src = _normalize_epub_href(seg.resource_href, src)
                        md_path = image_map.get(normalized_src, "")
                        if md_path:
                            blocks.append(f"![{alt}]({md_path})")

            if not seg.target:
                position += 1
                continue

            # Get images for this segment
            seg_images = segment_images.get((ch.index, seg.index), [])

            # Separate placement types: before/after/block (block-level) vs inline
            before_images = []
            after_images = []
            inline_images = []
            for inline_id, norm_src, alt, placement in seg_images:
                if placement == "inline":
                    inline_images.append((inline_id, norm_src, alt, placement))
                elif placement == "before" or placement == "block":
                    # "block" is treated as "before" for compatibility
                    before_images.append((inline_id, norm_src, alt, placement))
                elif placement == "after":
                    after_images.append((inline_id, norm_src, alt, placement))
                else:
                    # Fallback: treat unknown as after
                    after_images.append((inline_id, norm_src, alt, placement))

            # Fail if inline images are present (not supported in Markdown)
            if inline_images and markdown:
                raise ValueError(
                    f"Chapter {ch.index}, Segment {seg.index}: Inline images within text "
                    f"are not supported in Markdown export. Found {len(inline_images)} inline "
                    f"image(s). Please use EPUB format or remove inline images."
                )

            # Add images before segment text
            for inline_id, norm_src, alt, placement in before_images:
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

            # Add images after segment text
            for inline_id, norm_src, alt, placement in after_images:
                md_path = image_map.get(norm_src, "")
                if md_path:
                    alt_text = alt if alt else "Image"
                    blocks.append(f"![{alt_text}]({md_path})")

            position += 1

        # Output any remaining image blocks at the end
        if markdown and position in image_blocks_by_position:
            for img_info in image_blocks_by_position[position]:
                src = img_info.get("src", "")
                alt = img_info.get("alt", "Image")
                if src and ch.segments:
                    first_href = ch.segments[0].resource_href
                    if first_href:
                        normalized_src = _normalize_epub_href(first_href, src)
                        md_path = image_map.get(normalized_src, "")
                        if md_path:
                            blocks.append(f"![{alt}]({md_path})")

        chapter_blocks.append("\n\n".join(blocks))

    # Write output file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chapter_blocks) + "\n")

    return out_path


# Plain text.
def _assemble_text(
    store: AssembleStore,
    source_path: str,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """Rebuild UTF-8 text by chapter and paragraph, optionally including bilingual source text."""
    return _assemble_plain_text(store, source_path, out_path, bilingual=bilingual, order=order)


# ── markdown ──────────────────────────────────────────────────────────────────
def _assemble_markdown(
    store: AssembleStore,
    source_path: str,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """Rebuild Markdown with # headings and optional bilingual source text."""
    return _assemble_plain_text(
        store, source_path, out_path, bilingual=bilingual, order=order, markdown=True
    )
