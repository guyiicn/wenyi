"""Test that prepare completes glossary-only initialization."""

import os
import tempfile

import pytest

from wenyi_core.config import Config
from wenyi_core.ingest.source_hash import source_sha256
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.storage.run_store import RunStore, translation_run_dir
from wenyi_core.llm.providers.fake import FakeClient


def test_prepare_completes_glossary_initialization(tmp_path):
    """Verify prepare populates chapters after glossary import creates minimal manifest."""
    # Create a simple text document
    input_file = tmp_path / "test.txt"
    input_file.write_text("Test Novel\n\nChapter One\n\nThe Force is strong.\n\nThe Jedi await.")

    state_dir = tmp_path / "state"
    state_dir.mkdir()

    config = Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "paths": {"state_dir": str(state_dir)},
            "pipeline": {"book_understanding": False},
        }
    )

    # Simulate glossary import creating minimal manifest
    # The title will be derived from the filename
    title = "test"
    run_dir = translation_run_dir(str(state_dir), title, "zh")
    store = RunStore(run_dir, create=True)

    source_hash = source_sha256(str(input_file))
    store.begin_initialization(source_hash)

    # Create minimal manifest with glossary_import_only marker
    minimal_manifest = {
        "fmt": "txt",
        "title": title,
        "source_lang": "en",
        "target_lang": "zh",
        "source_sha256": source_hash,
        "chapters": [],
        "glossary_import_only": True,
    }
    store.write_artifact("manifest.json", minimal_manifest)
    store.finish_initialization()

    # Verify minimal state created by glossary import
    assert store.exists()
    manifest = store.load_manifest()
    assert manifest["glossary_import_only"] is True
    assert len(manifest["chapters"]) == 0

    # Now run prepare - it should complete initialization
    def mock_handler(messages, tier, json_mode):
        return '{"language": "en"}'

    client = FakeClient(handler=mock_handler)
    orch = Orchestrator(config, client=client)

    result_store = orch.prepare_for_translation(str(input_file))

    # Verify chapters were added
    final_manifest = result_store.load_manifest()
    assert (
        "glossary_import_only" not in final_manifest
        or final_manifest.get("glossary_import_only") is None
    )
    assert len(final_manifest["chapters"]) > 0

    # Verify chapters are accessible
    chapter = result_store.load_chapter(0)
    assert chapter is not None
    assert len(chapter.segments) > 0
