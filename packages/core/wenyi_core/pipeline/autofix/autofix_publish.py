"""Publish a persisted Autofix index with optimistic target checks and resumable alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...events import ProgressFn
from ...review.models import ReviewOutcome, integer_index, text_hash
from ...review.run_store import ReviewRunStore
from ...storage.protocol import Storage

if TYPE_CHECKING:
    from ..annotate.annotations import AnnotationService
    from ..annotate.docx_styles import DocxStyleService


class AutofixPublisher:
    def __init__(self, annotations: AnnotationService, docx_styles: DocxStyleService):
        self._annotations = annotations
        self._docx_styles = docx_styles
    
    def _has_inline_emphasis(self, segment: Any) -> bool:
        """Check if segment has inline emphasis metadata that needs preservation."""
        from ...markup.contracts import INLINE_META_KEY
        
        # Check source inline metadata
        raw_inline = segment.meta.get(INLINE_META_KEY)
        if not isinstance(raw_inline, dict):
            return False
        
        nodes = raw_inline.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            return False
        
        # Count semantic emphasis nodes (em, strong, i, b, span with CSS emphasis)
        emphasis_count = sum(
            1
            for node in nodes
            if isinstance(node, dict) and node.get("tag") in ("i", "em", "strong", "b", "span")
        )
        
        return emphasis_count > 0
    
    def _apply_with_inline_preservation(
        self,
        segment: Any,
        before_text: str,
        after_text: str,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Apply autofix modification while preserving inline emphasis markers.
        
        Returns:
            (success, new_target, new_inline_metadata)
            success=False if inline preservation failed
        """
        from ...markup.inline_markers import encode_inline_markers, decode_inline_markers, InlineNode
        
        # Get current target_inline metadata
        target_inline = segment.meta.get("target_inline")
        if not isinstance(target_inline, dict):
            # No target_inline metadata, cannot preserve
            return False, "", None
        
        ranges = target_inline.get("ranges")
        if not isinstance(ranges, list):
            return False, "", None
        
        # Convert ranges back to InlineNode objects
        inline_nodes: list[InlineNode] = []
        for r in ranges:
            if not isinstance(r, dict):
                continue
            tag = r.get("tag")
            start = r.get("start")
            end = r.get("end")
            if isinstance(tag, str) and isinstance(start, int) and isinstance(end, int):
                marker_name = {"i": "I", "em": "EM", "strong": "STRONG", "b": "B"}.get(
                    tag, tag.upper()
                )
                inline_nodes.append(
                    InlineNode(tag=tag, start=start, end=end, marker_name=marker_name)
                )
        
        if not inline_nodes:
            # No inline nodes to preserve
            return False, "", None
        
        # For autofix, we need to apply the text modification while preserving inline markers
        # Strategy: compute the diff between before_text and after_text, then apply it
        # while adjusting marker positions
        
        # Simple case: check if this is a simple replacement/insertion/deletion
        # For now, we'll re-encode the after_text with adjusted marker positions
        
        # Calculate position mapping from before to after
        # For a simple replacement like "AB" -> "ABC" (insertion at end)
        # or "ABC" -> "AB" (deletion at end), we can adjust marker positions proportionally
        
        # However, for reliability, we'll use a different approach:
        # Parse the modification as an edit operation and shift markers accordingly
        
        # For the initial implementation, we'll handle the common case:
        # Text replacement that doesn't overlap with emphasis spans
        
        # First, try to find the changed region
        before_len = len(before_text)
        after_len = len(after_text)
        
        # Find common prefix and suffix
        prefix_len = 0
        while prefix_len < min(before_len, after_len) and before_text[prefix_len] == after_text[prefix_len]:
            prefix_len += 1
        
        suffix_len = 0
        while (suffix_len < min(before_len, after_len) - prefix_len and 
               before_text[before_len - 1 - suffix_len] == after_text[after_len - 1 - suffix_len]):
            suffix_len += 1
        
        # The change is from [prefix_len : before_len - suffix_len] -> [prefix_len : after_len - suffix_len]
        change_start = prefix_len
        change_end_before = before_len - suffix_len
        change_end_after = after_len - suffix_len
        
        # Adjust inline node positions
        adjusted_nodes: list[Any] = []
        for node in inline_nodes:
            # Check if this node is affected by the change
            if node.end <= change_start:
                # Node is before the change, keep as is
                adjusted_nodes.append(
                    InlineNode(tag=node.tag, start=node.start, end=node.end, marker_name=node.marker_name)
                )
            elif node.start >= change_end_before:
                # Node is after the change, shift by delta
                delta = (change_end_after - change_end_before)
                adjusted_nodes.append(
                    InlineNode(tag=node.tag, start=node.start + delta, end=node.end + delta, marker_name=node.marker_name)
                )
            else:
                # Node overlaps with the change region
                # For now, we'll fail-closed: don't apply autofix if it affects emphasis
                return False, "", None
        
        # Now encode after_text with adjusted markers
        try:
            encoded_after = self._encode_text_with_inline_nodes(after_text, adjusted_nodes)
        except Exception:
            return False, "", None
        
        # Decode the modified text
        try:
            clean_text, new_ranges, warnings = decode_inline_markers(
                encoded_after, inline_nodes
            )
        except Exception:
            # Decode failed
            return False, "", None
        
        # Check if decode had critical warnings
        if warnings:
            critical = any(
                keyword in str(w).lower()
                for w in warnings
                for keyword in ["unclosed", "mismatched", "unmatched", "lost"]
            )
            if critical:
                return False, "", None
        
        # Build new target_inline metadata
        new_inline = {
            "version": 1,
            "target_length": len(clean_text),
            "ranges": [
                {"start": start, "end": end, "tag": tag}
                for start, end, tag in new_ranges
            ],
            "source_node_count": target_inline.get("source_node_count", len(inline_nodes)),
            "target_node_count": len(new_ranges),
            "validation_warnings": warnings if warnings else [],
        }
        
        # Check if all inline nodes were preserved
        if len(new_ranges) == 0 and len(inline_nodes) > 0:
            new_inline["preservation_failed"] = True
            return False, "", None
        
        return True, clean_text, new_inline
    
    def _encode_text_with_inline_nodes(
        self, text: str, nodes: list[Any]
    ) -> str:
        """Encode text with inline markers at specified positions."""
        # Sort nodes by position (reverse order for insertion)
        sorted_nodes = sorted(nodes, key=lambda n: n.start, reverse=True)
        
        result = text
        for node in sorted_nodes:
            # Insert end marker
            result = result[:node.end] + f"[[{node.marker_name}_END]]" + result[node.end:]
            # Insert start marker
            result = result[:node.start] + f"[[{node.marker_name}_START]]" + result[node.start:]
        
        return result

    def apply(
        self,
        store: Storage,
        debug: ReviewRunStore,
        index: dict[str, Any],
        *,
        progress: ProgressFn | None,
    ) -> None:
        """Publish formal targets idempotently from the index, then refresh character-offset
        metadata.
        """
        raw_locations = index.get("locations")
        locations = [row for row in raw_locations or [] if isinstance(row, dict)]
        by_chapter: dict[int, list[dict[str, Any]]] = {}
        for row in locations:
            chapter = integer_index(row.get("chapter"))
            if chapter is not None:
                by_chapter.setdefault(chapter, []).append(row)

        total = len(locations)
        done = 0
        if progress and total:
            progress(0, total, "Publishing Review Autofix")
        for chapter_index in sorted(by_chapter):
            chapter = store.load_chapter(chapter_index)
            chapter_locations = sorted(
                by_chapter[chapter_index], key=lambda row: row.get("index", -1)
            )
            applied_positions: list[int] = []
            chapter_changed = False
            for row in chapter_locations:
                text_index = row.get("index")
                target = row.get("target")
                before = row.get("before")
                if (
                    isinstance(text_index, bool)
                    or not isinstance(text_index, int)
                    or not 0 <= text_index < len(chapter.text_segments)
                    or not isinstance(target, str)
                    or not isinstance(before, str)
                ):
                    row["status"] = "failed"
                    row["reason"] = "invalid_index_location"
                    done += 1
                    continue
                segment = chapter.text_segments[text_index]
                current = segment.target or ""
                if current == target:
                    row["status"] = "no_net_change" if current == before else "applied"
                    if current != before and row.get("alignment_status") != "completed":
                        applied_positions.append(text_index)
                elif current == before and text_hash(current) == row.get("before_hash"):
                    # Check if segment has inline emphasis metadata
                    has_inline = self._has_inline_emphasis(segment)
                    
                    if has_inline:
                        # Apply modification through encode→edit→decode to preserve inline markers
                        success, new_target, new_inline = self._apply_with_inline_preservation(
                            segment, before, target
                        )
                        if success:
                            segment.target = new_target
                            if new_inline:
                                segment.meta["target_inline"] = new_inline
                            row["status"] = "applied"
                            applied_positions.append(text_index)
                            chapter_changed = True
                        else:
                            # Inline preservation failed during autofix
                            row["status"] = "failed"
                            row["reason"] = "inline_preservation_failed_during_autofix"
                    else:
                        # No inline emphasis, apply normally
                        segment.target = target
                        row["status"] = "applied"
                        applied_positions.append(text_index)
                        chapter_changed = True
                else:
                    row["status"] = "failed"
                    row["reason"] = "formal_target_changed"
                    row["actual_hash"] = text_hash(current)
                done += 1
                if progress:
                    progress(done, total, "Publishing Review Autofix")
            if chapter_changed:
                store.save_chapter(chapter)

            # Post-translation alignment depends on target offsets; refresh it against final published text.
            for text_index in sorted(set(applied_positions)):
                row = next(item for item in chapter_locations if item.get("index") == text_index)
                if row.get("status") == "failed" or row.get("alignment_status") == "completed":
                    continue
                self._annotations.align_annotations_after_batch(
                    chapter_index,
                    chapter,
                    text_index,
                    1,
                    store,
                )
                self._docx_styles.align_styles_after_batch(
                    chapter_index,
                    chapter,
                    text_index,
                    1,
                    store,
                )
                row["alignment_status"] = "completed"
            debug.write_json("autofix/index.json", index)

        location_status = {
            (row.get("chapter"), row.get("index")): row.get("status") for row in locations
        }
        for record in index.get("records", []):
            if not isinstance(record, dict) or record.get("status") != "planned":
                continue
            status = location_status.get((record.get("chapter"), record.get("index")))
            if status == "applied":
                record["status"] = "applied"
            elif status == "no_net_change":
                record["status"] = "not_applied_no_net_change"
            elif status == "failed":
                record["status"] = "not_applied"

        has_failures = any(
            isinstance(record, dict) and record.get("status") in {"failed", "not_applied"}
            for record in index.get("records", [])
        ) or any(row.get("status") == "failed" for row in locations)
        index["status"] = "partial" if has_failures else "completed"
        debug.write_json("autofix/index.json", index)

    def finish(
        self,
        store: Storage,
        debug: ReviewRunStore,
        index: dict[str, Any],
        result: dict[str, Any],
    ) -> ReviewOutcome:
        """Merge the autofix publication summary into review result.json idempotently."""
        records = [record for record in index.get("records", []) if isinstance(record, dict)]
        locations = [row for row in index.get("locations", []) if isinstance(row, dict)]
        applied_locations = [row for row in locations if row.get("status") == "applied"]
        failed_issue_records = [
            record
            for record in records
            if record.get("origin") == "final_issue_fix"
            and record.get("status") in {"failed", "not_applied"}
        ]
        autofix = {
            "enabled": True,
            "status": index.get("status", "partial"),
            "index": "autofix/index.json",
            "applied_segment_count": len(applied_locations),
            "applied_change_count": sum(
                record.get("origin") == "change" and record.get("status") == "applied"
                for record in records
            ),
            "applied_issue_fix_count": sum(
                record.get("origin") == "final_issue_fix" and record.get("status") == "applied"
                for record in records
            ),
            "failed_issue_count": sum(
                max(1, len(record.get("issue_ids") or [])) for record in failed_issue_records
            ),
            "failed_record_count": sum(
                record.get("status") in {"failed", "not_applied"} for record in records
            ),
        }
        updated = dict(result)
        existing_autofix = updated.get("autofix")
        if (
            isinstance(existing_autofix, dict)
            and existing_autofix.get("index") == "autofix/index.json"
            and existing_autofix.get("status") == autofix["status"]
        ):
            return ReviewOutcome(
                run_dir=debug.run_dir,
                result=updated,
                usage=debug.load_usage() or {},
            )
        updated["summary"] = {
            **dict(result.get("summary") or {}),
            "autofix_applied_segment_count": autofix["applied_segment_count"],
            "autofix_failed_issue_count": autofix["failed_issue_count"],
        }
        updated["autofix"] = autofix
        debug.write_json("result.json", updated)
        debug.log_event("review_autofix_finished", **autofix)
        store.log_event(
            "review_autofix_finished",
            review_id=debug.review_id,
            status=autofix["status"],
            applied_segment_count=autofix["applied_segment_count"],
            failed_issue_count=autofix["failed_issue_count"],
        )
        return ReviewOutcome(
            run_dir=debug.run_dir,
            result=updated,
            usage=debug.load_usage() or {},
        )
