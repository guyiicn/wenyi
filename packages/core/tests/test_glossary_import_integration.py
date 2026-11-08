"""Integration tests for glossary import reaching translation and review prompts."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from wenyi_core.config import Config
from wenyi_core.glossary.store import GlossaryStore
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.models.glossary import GlossaryTerm
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.file import FileStorage
from wenyi_core.llm.providers.fake import FakeClient


def _single_chapter_txt(path: str) -> None:
    """Write a minimal single-chapter text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("Test Novel\n\nChapter 1\n\nThe Force is strong. The Jedi must prevail.")


class TestGlossaryImportPrompts(unittest.TestCase):
    """Verify imported glossary reaches translator and reviewer prompts."""

    def test_imported_glossary_reaches_translator_prompt(self):
        """Imported terms appear in translation prompts without terminology_context."""
        with tempfile.TemporaryDirectory() as tmp:
            txt = os.path.join(tmp, "novel.txt")
            _single_chapter_txt(txt)

            config = Config.from_dict(
                {
                    "language": {"source": "en", "target": "zh"},
                    "paths": {"state_dir": os.path.join(tmp, "state")},
                    "pipeline": {
                        "book_understanding": False,  # Skip prescan for speed
                        "polish": False,
                        "terminology_context": False,  # Explicitly off
                    },
                }
            )

            captured_prompts = []

        def capture_handler(messages, tier, json_mode):
            captured_prompts.append(messages)
            if "literary translator" in messages[0]["content"]:
                return '{"translations": ["原力很强。绝地必将获胜。"]}'
            if "title" in messages[0]["content"].lower() or "title" in messages[-1]["content"].lower():
                return '{"titles": ["小说"]}'
            return '{"language": "en"}'

            client = FakeClient(handler=capture_handler)
            orch = Orchestrator(config, client=client)

            # Initialize state
            store = orch.prepare(txt)

            # Import glossary
            glossary = GlossaryStore(store.glossary_path)
            try:
                glossary.upsert_term(
                    GlossaryTerm(
                        source="Force",
                        target="原力",
                        type="term",
                        note="Star Wars concept",
                        status="ok",
                    )
                )
                glossary.upsert_term(
                    GlossaryTerm(source="Jedi", target="绝地", type="term", status="provisional")
                )
            finally:
                glossary.close()

            # Translate
            orch.run(txt, only_chapter=0)

            # Find translation prompt
            translation_prompts = [
                msg
                for msg in captured_prompts
                if any("literary translator" in m.get("content", "") for m in msg)
            ]
            self.assertGreater(len(translation_prompts), 0)

            # Verify glossary is in prompt
            user_content = translation_prompts[0][-1]["content"]
            self.assertIn("Force", user_content)
            self.assertIn("原力", user_content)
            self.assertIn("Jedi", user_content)
            self.assertIn("绝地", user_content)
            # Notes are only shown when terminology_context is on, but terms are always shown
            self.assertNotIn("Star Wars concept", user_content)

    def test_imported_glossary_reaches_reviewer_prompt(self):
        """Imported terms appear in review prompts."""
        with tempfile.TemporaryDirectory() as tmp:
            txt = os.path.join(tmp, "novel.txt")
            _single_chapter_txt(txt)

            config = Config.from_dict(
                {
                    "language": {"source": "en", "target": "zh"},
                    "paths": {"state_dir": os.path.join(tmp, "state")},
                    "pipeline": {
                        "book_understanding": False,
                        "polish": False,
                        "review": True,
                        "terminology_context": False,
                    },
                }
            )

            captured_prompts = []

        def capture_handler(messages, tier, json_mode):
            captured_prompts.append(messages)
            if "literary translator" in messages[0]["content"]:
                return '{"translations": ["原力很强。绝地必将获胜。"]}'
            if "translation reviewer" in messages[0]["content"]:
                return '{"issues": []}'
            if "title" in messages[0]["content"].lower() or "title" in messages[-1]["content"].lower():
                return '{"titles": ["小说"]}'
            return '{"language": "en"}'

            client = FakeClient(handler=capture_handler)
            orch = Orchestrator(config, client=client)

            # Initialize and import
            store = orch.prepare(txt)
            glossary = GlossaryStore(store.glossary_path)
            try:
                glossary.upsert_term(GlossaryTerm(source="Force", target="原力", type="term"))
                glossary.upsert_term(GlossaryTerm(source="Jedi", target="绝地", type="term"))
            finally:
                glossary.close()

            # Translate and review
            orch.run_all(txt)

            # Find review prompt
            review_prompts = [
                msg
                for msg in captured_prompts
                if any("translation reviewer" in m.get("content", "") for m in msg)
            ]
            self.assertGreater(len(review_prompts), 0)

            # Verify glossary is in review prompt
            user_content = review_prompts[0][-1]["content"]
            self.assertIn("Force", user_content)
            self.assertIn("原力", user_content)
            self.assertIn("Jedi", user_content)
            self.assertIn("绝地", user_content)

    def test_only_chapter_sends_single_chapter_source(self):
        """Using --chapter with imported glossary sends only that chapter's source."""
        with tempfile.TemporaryDirectory() as tmp:
            # Two-chapter book
            txt = os.path.join(tmp, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(
                    "Two Chapters\n\n"
                    "Chapter 1\n\n"
                    "First chapter text about Jedi.\n\n"
                    "Chapter 2\n\n"
                    "Second chapter text about Sith."
                )

            config = Config.from_dict(
                {
                    "language": {"source": "en", "target": "zh"},
                    "paths": {"state_dir": os.path.join(tmp, "state")},
                    "pipeline": {"book_understanding": False, "polish": False},
                }
            )

            captured_sources = []

            def capture_handler(messages, tier, json_mode):
                if "literary translator" in messages[0]["content"]:
                    user = messages[-1]["content"]
                    # Extract source text
                    if "[0]" in user:
                        start = user.index("[0]")
                        end = user.find("\n\n", start) if "\n\n" in user[start:] else len(user)
                        captured_sources.append(user[start:end])
                    return '{"translations": ["第一章关于绝地的文字。"]}'
                return '{"language": "en"}'

            client = FakeClient(handler=capture_handler)
            orch = Orchestrator(config, client=client)

            # Initialize and import
            store = orch.prepare(txt)
            glossary = GlossaryStore(store.glossary_path)
            try:
                glossary.upsert_term(GlossaryTerm(source="Jedi", target="绝地"))
                glossary.upsert_term(GlossaryTerm(source="Sith", target="西斯"))
            finally:
                glossary.close()

            # Translate only chapter 0
            orch.run(txt, only_chapter=0)

            # Verify only chapter 0 source was sent
            self.assertGreater(len(captured_sources), 0)
            combined = " ".join(captured_sources)
            self.assertIn("Jedi", combined)
            self.assertNotIn("Sith", combined)  # Chapter 1 was not sent


class TestGlossaryStorePublicInterface(unittest.TestCase):
    """Verify GlossaryStore public methods support import use case."""

    def test_upsert_term_public_interface(self):
        """upsert_term is public and documented."""
        with tempfile.TemporaryDirectory() as tmp:
            store = GlossaryStore(os.path.join(tmp, "glossary.db"))
            try:
                # Public method exists
                self.assertTrue(callable(getattr(store, "upsert_term", None)))

                # Accepts GlossaryTerm and optional chapter
                result = store.upsert_term(
                    GlossaryTerm(source="Test", target="测试", type="term"), chapter=0
                )
                self.assertEqual(result, "inserted")

                # Returns outcome string
                result2 = store.upsert_term(
                    GlossaryTerm(source="Test", target="测试", type="term"), chapter=0
                )
                self.assertEqual(result2, "unchanged")
            finally:
                store.close()

    def test_all_terms_preserves_import_order(self):
        """all_terms returns terms in insertion order for prompt stability."""
        with tempfile.TemporaryDirectory() as tmp:
            store = GlossaryStore(os.path.join(tmp, "glossary.db"))
            try:
                store.upsert_term(GlossaryTerm(source="Zulu", target="祖鲁"))
                store.upsert_term(GlossaryTerm(source="Alpha", target="阿尔法"))
                store.upsert_term(GlossaryTerm(source="Mike", target="迈克"))

                terms = store.all_terms()
                sources = [t.source for t in terms]
                # Insertion order, not alphabetical
                self.assertEqual(sources, ["Zulu", "Alpha", "Mike"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
