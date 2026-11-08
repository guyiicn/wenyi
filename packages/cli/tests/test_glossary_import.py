"""Test glossary import from external TSV files."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rich.console import Console
from wenyi_core.config import Config
from wenyi_core.glossary.store import GlossaryStore
from wenyi_core.storage.file import FileStorage

from wenyi_cli.commands.glossary_import import import_glossary


class TestGlossaryImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.tmp.name) / "state"
        self.state_dir.mkdir()
        self.console = Console(file=open(os.devnull, "w"))

    def tearDown(self):
        self.tmp.cleanup()

    def _init_store(self) -> FileStorage:
        """Initialize a minimal translation state."""
        config = Config.from_dict(
            {"language": {"source": "en", "target": "zh"}, "paths": {"state_dir": str(self.state_dir)}}
        )
        run_dir = self.state_dir / "test-novel" / "targets" / "zh"
        run_dir.mkdir(parents=True)

        store = FileStorage(str(run_dir), create=True)
        # Minimal manifest
        manifest = {
            "fmt": "txt",
            "title": "Test Novel",
            "source_lang": "en",
            "target_lang": "zh",
            "source_sha256": "abc123",
            "chapters": [],
        }
        store.save_manifest(manifest)
        return store

    def _write_tsv(self, filename: str, content: str) -> Path:
        """Write TSV content and return its path."""
        path = Path(self.tmp.name) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_import_basic_terms(self):
        """Import basic terms with source and target."""
        store = self._init_store()
        tsv = self._write_tsv(
            "terms.tsv",
            "english\tpreferred_zh_cn\n"
            "Force\t原力\n"
            "Jedi\t绝地\n"
            "lightsaber\t光剑\n",
        )

        results = import_glossary(
            store, tsv, source_col="english", target_col="preferred_zh_cn", console=self.console
        )

        self.assertEqual(results["inserted"], 3)
        self.assertEqual(results["unchanged"], 0)
        self.assertEqual(results["conflict"], 0)

        glossary = GlossaryStore(store.glossary_path)
        try:
            terms = glossary.all_terms()
            self.assertEqual(len(terms), 3)
            force = glossary.get_term("Force")
            self.assertIsNotNone(force)
            self.assertEqual(force.target, "原力")
        finally:
            glossary.close()

    def test_import_with_type_and_aliases(self):
        """Import terms with type and aliases."""
        store = self._init_store()
        tsv = self._write_tsv(
            "terms.tsv",
            "english\tpreferred_zh_cn\ttype\taliases_and_regional\n"
            "Rey\t蕾伊\tperson\tRey Skywalker\n"
            "BB-8\tBB-8\tdroid\t\n",
        )

        results = import_glossary(
            store,
            tsv,
            source_col="english",
            target_col="preferred_zh_cn",
            type_col="type",
            aliases_col="aliases_and_regional",
            console=self.console,
        )

        self.assertEqual(results["inserted"], 2)
        glossary = GlossaryStore(store.glossary_path)
        try:
            rey = glossary.get_term("Rey")
            self.assertIsNotNone(rey)
            self.assertEqual(rey.type, "person")
            self.assertEqual(rey.aliases, ["Rey Skywalker"])

            bb8 = glossary.get_term("BB-8")
            self.assertIsNotNone(bb8)
            self.assertEqual(bb8.type, "term")  # droid maps to term
            self.assertEqual(bb8.aliases, [])
        finally:
            glossary.close()

    def test_import_with_status_and_note(self):
        """Import terms with status and usage notes."""
        store = self._init_store()
        tsv = self._write_tsv(
            "terms.tsv",
            "english\tpreferred_zh_cn\tstatus\tusage_conflicts\n"
            "the Force\t原力\tverified\t仅用于星战概念\n"
            "Grogu\t古古\tprovisional\t待确认\n",
        )

        results = import_glossary(
            store,
            tsv,
            source_col="english",
            target_col="preferred_zh_cn",
            status_col="status",
            note_col="usage_conflicts",
            console=self.console,
        )

        self.assertEqual(results["inserted"], 2)
        glossary = GlossaryStore(store.glossary_path)
        try:
            force = glossary.get_term("the Force")
            self.assertIsNotNone(force)
            self.assertEqual(force.status, "ok")  # verified maps to ok
            self.assertEqual(force.note, "仅用于星战概念")

            grogu = glossary.get_term("Grogu")
            self.assertIsNotNone(grogu)
            self.assertEqual(grogu.status, "provisional")
            self.assertEqual(grogu.note, "待确认")
        finally:
            glossary.close()

    def test_import_idempotent_same_data(self):
        """Re-importing identical data reports unchanged."""
        store = self._init_store()
        tsv = self._write_tsv("terms.tsv", "english\tpreferred_zh_cn\nJedi\t绝地\n")

        # First import
        results1 = import_glossary(
            store, tsv, source_col="english", target_col="preferred_zh_cn", console=self.console
        )
        self.assertEqual(results1["inserted"], 1)

        # Second import
        results2 = import_glossary(
            store, tsv, source_col="english", target_col="preferred_zh_cn", console=self.console
        )
        self.assertEqual(results2["inserted"], 0)
        self.assertEqual(results2["unchanged"], 1)

    def test_import_conflict_preserves_existing(self):
        """Conflicting target is recorded but does not overwrite existing."""
        store = self._init_store()
        glossary = GlossaryStore(store.glossary_path)
        try:
            from wenyi_core.models.glossary import GlossaryTerm

            # Pre-populate existing term
            glossary.upsert_term(GlossaryTerm(source="Jedi", target="绝地武士"), chapter=0)
        finally:
            glossary.close()

        tsv = self._write_tsv("terms.tsv", "english\tpreferred_zh_cn\nJedi\t绝地\n")
        results = import_glossary(
            store, tsv, source_col="english", target_col="preferred_zh_cn", console=self.console
        )

        self.assertEqual(results["conflict"], 1)
        glossary = GlossaryStore(store.glossary_path)
        try:
            jedi = glossary.get_term("Jedi")
            self.assertIsNotNone(jedi)
            self.assertEqual(jedi.target, "绝地武士")  # Existing preserved
            conflicts = glossary.open_conflicts()
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["source"], "Jedi")
        finally:
            glossary.close()

    def test_import_dry_run_no_writes(self):
        """Dry-run mode does not write to glossary."""
        store = self._init_store()
        tsv = self._write_tsv("terms.tsv", "english\tpreferred_zh_cn\nForce\t原力\n")

        results = import_glossary(
            store,
            tsv,
            source_col="english",
            target_col="preferred_zh_cn",
            dry_run=True,
            console=self.console,
        )

        self.assertEqual(results["inserted"], 1)
        glossary = GlossaryStore(store.glossary_path)
        try:
            terms = glossary.all_terms()
            self.assertEqual(len(terms), 0)  # Nothing written
        finally:
            glossary.close()

    def test_import_rejects_duplicate_source(self):
        """TSV with duplicate source entries is rejected."""
        store = self._init_store()
        tsv = self._write_tsv(
            "terms.tsv",
            "english\tpreferred_zh_cn\n" "Jedi\t绝地\n" "Jedi\t绝地武士\n",
        )

        from wenyi_cli.commands.glossary_import import ImportError

        with self.assertRaises(ImportError) as ctx:
            import_glossary(
                store, tsv, source_col="english", target_col="preferred_zh_cn", console=self.console
            )

        self.assertIn("duplicate source", str(ctx.exception).lower())

    def test_import_skips_empty_rows(self):
        """Rows with empty source or target are skipped."""
        store = self._init_store()
        tsv = self._write_tsv(
            "terms.tsv",
            "english\tpreferred_zh_cn\n"
            "\t原力\n"  # Empty source
            "Jedi\t\n"  # Empty target
            "lightsaber\t光剑\n",  # Valid
        )

        results = import_glossary(
            store, tsv, source_col="english", target_col="preferred_zh_cn", console=self.console
        )

        self.assertEqual(results["inserted"], 1)
        glossary = GlossaryStore(store.glossary_path)
        try:
            terms = glossary.all_terms()
            self.assertEqual(len(terms), 1)
            self.assertEqual(terms[0].source, "lightsaber")
        finally:
            glossary.close()

    def test_import_missing_column_error(self):
        """Missing required column raises clear error."""
        store = self._init_store()
        tsv = self._write_tsv("terms.tsv", "source\ttarget\nJedi\t绝地\n")

        from wenyi_cli.commands.glossary_import import ImportError

        with self.assertRaises(ImportError) as ctx:
            import_glossary(
                store, tsv, source_col="english", target_col="preferred_zh_cn", console=self.console
            )

        self.assertIn("english", str(ctx.exception).lower())

    def test_import_no_state_error(self):
        """Import before state initialization raises error."""
        run_dir = self.state_dir / "nonexistent" / "targets" / "zh"
        run_dir.mkdir(parents=True)
        store = FileStorage(str(run_dir), create=False)

        tsv = self._write_tsv("terms.tsv", "english\tpreferred_zh_cn\nJedi\t绝地\n")

        from wenyi_cli.commands.glossary_import import ImportError

        with self.assertRaises(ImportError) as ctx:
            import_glossary(
                store, tsv, source_col="english", target_col="preferred_zh_cn", console=self.console
            )

        self.assertIn("no translation state", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
