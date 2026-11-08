"""Explicit batch inputs and translate/polish results without formal state mutation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from ...agents.polisher import Polisher
from ...agents.translator import Translator
from ...ingest.segmenter import batch_segments
from ...markup.ruby import strip_ruby_markers
from ...models.document import Segment
from ...models.glossary import GlossaryTerm


@dataclass(frozen=True)
class BatchPlan:
    """Capture the source positions and context used for exactly one serial batch."""

    chapter: int
    start_index: int
    segment_indices: tuple[int, ...]
    sources: tuple[str, ...]
    terms: tuple[GlossaryTerm, ...]
    context: str
    style: str
    book_synopsis: str
    chapter_digest: str
    annotation_contexts: list[list[dict[str, str]]]
    next_source: str
    allow_empty_translations: bool = False
    segment_metas: tuple[dict[str, object], ...] = ()

    @classmethod
    def capture(
        cls,
        chapter: int,
        start_index: int,
        segments: list[Segment],
        terms: list[GlossaryTerm],
        context: str,
        style: str,
        book_synopsis: str,
        chapter_digest: str,
        annotation_contexts: list[list[dict[str, str]]],
        next_source: str,
        *,
        allow_empty_translations: bool = False,
    ) -> BatchPlan:
        """Detach mutable term and annotation metadata from subsequent chapter updates."""
        return cls(
            chapter,
            start_index,
            tuple(s.index for s in segments),
            tuple(s.source for s in segments),
            tuple(deepcopy(terms)),
            context,
            style,
            book_synopsis,
            chapter_digest,
            deepcopy(annotation_contexts),
            next_source,
            allow_empty_translations,
            tuple(deepcopy(s.meta) for s in segments),
        )


@dataclass(frozen=True)
class BatchResult:
    targets: tuple[str, ...]
    before_polish: tuple[str | None, ...]
    precision_key: str | None = None
    inline_metas: tuple[dict[str, object], ...] = ()


class TranslationBatchExecutor:
    def __init__(self, translator: Translator, polisher: Polisher):
        self._translator = translator
        self._polisher = polisher

    def execute(self, plan: BatchPlan, *, polish: bool) -> BatchResult:
        """Translate and optionally polish; the chapter service alone applies the result."""
        terms = list(plan.terms)
        targets = self._translator.translate_batch(
            list(plan.sources),
            glossary_terms=terms,
            style=plan.style,
            context=plan.context,
            book_synopsis=plan.book_synopsis,
            chapter_digest=plan.chapter_digest,
            annotation_contexts=plan.annotation_contexts,
            next_source=plan.next_source,
            allow_empty_translations=plan.allow_empty_translations,
            segment_metas=list(plan.segment_metas) if plan.segment_metas else None,
        )
        targets = [strip_ruby_markers(target) for target in targets]
        inline_metas = self._translator.last_inline_metas or []
        before_polish: tuple[str | None, ...] = (None,) * len(targets)
        if polish:
            from ...markup.inline_markers import InlineNode
            from ..translate.inline_translation import (
                decode_targets_with_inline,
                inline_preservation_prompt_suffix,
            )

            before_polish = tuple(targets)

            # Encode inline markers before polish
            encoded_targets: list[str] = []
            inline_nodes_per_target: list[list[InlineNode] | None] = []

            for target, inline_meta in zip(
                targets, inline_metas if inline_metas else [{}] * len(targets)
            ):
                if not isinstance(inline_meta, dict):
                    encoded_targets.append(target)
                    inline_nodes_per_target.append(None)
                    continue

                ranges = inline_meta.get("ranges")
                if not isinstance(ranges, list) or not ranges:
                    encoded_targets.append(target)
                    inline_nodes_per_target.append(None)
                    continue

                # Convert ranges to InlineNode objects for encoding
                nodes = []
                for r in ranges:
                    if not isinstance(r, dict):
                        continue
                    tag = r.get("tag")
                    start = r.get("start")
                    end = r.get("end")
                    css_style = r.get("css_style")
                    if isinstance(tag, str) and isinstance(start, int) and isinstance(end, int):
                        marker_name = {
                            "i": "I",
                            "em": "EM",
                            "strong": "STRONG",
                            "b": "B",
                            "span": "SPAN",
                        }.get(tag, tag.upper())
                        nodes.append(
                            InlineNode(
                                tag=tag,
                                start=start,
                                end=end,
                                marker_name=marker_name,
                                css_style=css_style if isinstance(css_style, str) else None,
                            )
                        )

                if nodes:
                    # Insert markers in reverse order
                    encoded = target
                    for node in sorted(nodes, key=lambda n: n.start, reverse=True):
                        marker = f"[[{node.marker_name}_END]]"
                        encoded = encoded[: node.end] + marker + encoded[node.end :]
                        marker = f"[[{node.marker_name}_START]]"
                        encoded = encoded[: node.start] + marker + encoded[node.start :]
                    encoded_targets.append(encoded)
                    inline_nodes_per_target.append(nodes)
                else:
                    encoded_targets.append(target)
                    inline_nodes_per_target.append(None)

            has_inline = any(nodes is not None for nodes in inline_nodes_per_target)

            # Polish with inline preservation if needed
            turn = self._translator.last_batch_turn
            indices = self._translator.last_batch_indices
            polished: list[str] | None = None

            # Monkey-patch polisher render safely with try/finally
            original_render = self._polisher.render if has_inline else None

            try:
                if has_inline:

                    def render_with_suffix(template_name, **kwargs):
                        result = original_render(template_name, **kwargs)
                        if template_name in ("polisher_user", "polisher_continue_user"):
                            result += inline_preservation_prompt_suffix()
                        return result

                    self._polisher.render = render_with_suffix

                if turn is not None and indices is not None:
                    # Continue with encoded targets (keep markers for decode)
                    continued = self._polisher.polish_continue(
                        turn,
                        n=len(indices),
                        next_source=plan.next_source,
                    )
                    if continued is not None and len(continued) == len(indices):
                        polished = list(encoded_targets)
                        for index, text in zip(indices, continued):
                            polished[index] = text  # Keep markers, will decode later

                if polished is None:
                    polished = self._polisher.polish(
                        encoded_targets,
                        glossary_terms=terms,
                        style=plan.style,
                        next_source=plan.next_source,
                        sources=list(plan.sources),
                    )
            finally:
                # Always restore original render to prevent pollution across batches
                if original_render is not None:
                    self._polisher.render = original_render

            # Decode inline markers from polished text
            if len(polished) == len(targets):
                # Strip ruby markers first, then decode inline
                polished = [strip_ruby_markers(text) for text in polished]
                clean_polished, new_inline_metas, warnings = decode_targets_with_inline(
                    polished, inline_nodes_per_target
                )

                if warnings:
                    # Log warnings but continue (polish may adjust text legitimately)
                    pass

                targets = clean_polished
                inline_metas = list(new_inline_metas)

        return BatchResult(tuple(targets), before_polish, None, tuple(inline_metas))


def resume_batches(segments: list[Segment], max_tokens: int) -> list[list[Segment]]:
    """Split token-budget batches again at completed/pending boundaries.
    A changed budget may mix saved translations and unset targets in one batch. Group by
    completion state to translate only missing paragraphs and avoid overwriting confirmed
    content. A saved blank string is complete; only None means pending.
    """
    batches: list[list[Segment]] = []
    for raw_batch in batch_segments(segments, max_tokens):
        current: list[Segment] = []
        current_done: bool | None = None
        for segment in raw_batch:
            done = segment.target is not None
            if current and done != current_done:
                batches.append(current)
                current = []
            current.append(segment)
            current_done = done
        if current:
            batches.append(current)
    return batches
