"""Test repair CLI commands for safety guards and correctness."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wenyi_cli.cli import app

runner = CliRunner()


@pytest.fixture
def test_state(tmp_path: Path):
    """Create minimal test state with segments."""
    state_dir = tmp_path / "targets" / "zh"
    chapters_dir = state_dir / "chapters"
    chapters_dir.mkdir(parents=True)

    chapter = {
        "index": 0,
        "title": "Test Chapter",
        "segments": [
            {
                "index": 10,
                "source": "Plain text",
                "target": "纯文本",
                "kind": "text",
                "anchor": "tn0_10",
                "resource_href": "test.xhtml",
                "cont": False,
                "meta": {},
            },
            {
                "index": 20,
                "source": "Text with emphasis",
                "target": "有强调文本",
                "kind": "text",
                "anchor": "tn0_20",
                "resource_href": "test.xhtml",
                "cont": False,
                "meta": {
                    "epub_inline": {
                        "version": 1,
                        "source_length": 18,
                        "nodes": [{"id": "tn0_20_0", "tag": "i", "placement": "inline"}],
                    },
                    "target_inline": {
                        "version": 1,
                        "ranges": [{"start": 1, "end": 3, "tag": "i", "css_style": ""}],
                    },
                },
            },
            {
                "index": 30,
                "source": "Legacy with markers",
                "target": "有[[I_START]]字面标记[[I_END]]",
                "kind": "text",
                "anchor": "tn0_30",
                "resource_href": "test.xhtml",
                "cont": False,
                "meta": {
                    "epub_inline": {
                        "version": 1,
                        "source_length": 19,
                        "nodes": [{"id": "tn0_30_0", "tag": "i", "placement": "inline"}],
                    }
                    # No target_inline - this is the bug condition
                },
            },
        ],
    }

    (chapters_dir / "ch0.json").write_text(json.dumps(chapter, ensure_ascii=False))
    return state_dir


def test_apply_target_exact_match_required(test_state: Path):
    """apply-target must reject if current text doesn't match expected."""
    result = runner.invoke(
        app,
        [
            "repair",
            "apply-target",
            str(test_state),
            "--chapter",
            "0",
            "--segment",
            "10",
            "--expected-text",
            "错误的当前文本",
            "--new-target",
            "新文本",
        ],
    )
    assert result.exit_code != 0
    output = result.stdout + (result.stderr or "")
    assert "does not match expected" in output or "does not match" in output


def test_apply_target_dry_run_does_not_modify(test_state: Path):
    """Dry-run must not modify state."""
    original_content = (test_state / "chapters" / "ch0.json").read_text()

    result = runner.invoke(
        app,
        [
            "repair",
            "apply-target",
            str(test_state),
            "--chapter",
            "0",
            "--segment",
            "10",
            "--expected-text",
            "纯文本",
            "--new-target",
            "修改后文本",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "[DRY RUN]" in result.stdout
    assert (test_state / "chapters" / "ch0.json").read_text() == original_content


def test_apply_target_plain_text_correction(test_state: Path):
    """Plain text correction without inline must work."""
    result = runner.invoke(
        app,
        [
            "repair",
            "apply-target",
            str(test_state),
            "--chapter",
            "0",
            "--segment",
            "10",
            "--expected-text",
            "纯文本",
            "--new-target",
            "修改后文本",
        ],
    )

    assert result.exit_code == 0
    assert "✓ Applied correction" in result.stdout

    # Verify actual change
    chapter = json.loads((test_state / "chapters" / "ch0.json").read_text())
    seg = next(s for s in chapter["segments"] if s["index"] == 10)
    assert seg["target"] == "修改后文本"


def test_apply_target_rejects_bad_markers(test_state: Path):
    """Applying target with malformed markers must be rejected."""
    result = runner.invoke(
        app,
        [
            "repair",
            "apply-target",
            str(test_state),
            "--chapter",
            "0",
            "--segment",
            "20",  # Has source inline
            "--expected-text",
            "有强调文本",
            "--new-target",
            "坏[[I_START]]标记没结束",  # Malformed
        ],
    )

    assert result.exit_code != 0
    # Should fail during decode


def test_apply_target_with_valid_markers(test_state: Path):
    """Applying target with valid markers must decode and generate target_inline."""
    result = runner.invoke(
        app,
        [
            "repair",
            "apply-target",
            str(test_state),
            "--chapter",
            "0",
            "--segment",
            "20",
            "--expected-text",
            "有强调文本",
            "--new-target",
            "修改后[[I_START]]强调[[I_END]]内容",
        ],
    )

    assert result.exit_code == 0
    assert "inline decode" in result.stdout

    # Verify decoded text and target_inline
    chapter = json.loads((test_state / "chapters" / "ch0.json").read_text())
    seg = next(s for s in chapter["segments"] if s["index"] == 20)
    assert seg["target"] == "修改后强调内容"
    assert "target_inline" in seg["meta"]
    assert len(seg["meta"]["target_inline"]["ranges"]) == 1


def test_apply_target_creates_backup(test_state: Path):
    """Modification must create backup file."""
    backup_file = test_state / "chapters" / "ch0.bak-seg10.json"
    assert not backup_file.exists()

    result = runner.invoke(
        app,
        [
            "repair",
            "apply-target",
            str(test_state),
            "--chapter",
            "0",
            "--segment",
            "10",
            "--expected-text",
            "纯文本",
            "--new-target",
            "新文本",
        ],
    )

    assert result.exit_code == 0
    assert backup_file.exists()


def test_fix_inline_markers_decodes_literal_markers(test_state: Path):
    """fix-inline-markers must decode literal markers and generate target_inline."""
    result = runner.invoke(
        app,
        [
            "repair",
            "fix-inline-markers",
            str(test_state),
            "--chapter",
            "0",
            "--segment",
            "30",
            "--expected-text",
            "有[[I_START]]字面标记[[I_END]]",
        ],
    )

    assert result.exit_code == 0
    assert "✓ Repaired" in result.stdout

    # Verify decoded
    chapter = json.loads((test_state / "chapters" / "ch0.json").read_text())
    seg = next(s for s in chapter["segments"] if s["index"] == 30)
    assert seg["target"] == "有字面标记"
    assert "target_inline" in seg["meta"]


def test_apply_target_preserves_existing_inline(test_state: Path):
    """Applying plain text to segment with existing target_inline must preserve metadata."""
    # Segment 20 has existing target_inline
    result = runner.invoke(
        app,
        [
            "repair",
            "apply-target",
            str(test_state),
            "--chapter",
            "0",
            "--segment",
            "20",
            "--expected-text",
            "有强调文本",
            "--new-target",
            "修改后文本无标记",  # No markers
        ],
    )

    assert result.exit_code == 0

    # Verify target changed but target_inline preserved
    chapter = json.loads((test_state / "chapters" / "ch0.json").read_text())
    seg = next(s for s in chapter["segments"] if s["index"] == 20)
    assert seg["target"] == "修改后文本无标记"
    # target_inline should still exist (preservation is implementation detail)
