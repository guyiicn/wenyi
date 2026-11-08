"""Test Polish/Review inline marker preservation with text length and order changes."""

from __future__ import annotations

import json

from wenyi_core.agents.polisher import Polisher
from wenyi_core.agents.translator import Translator
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.translate.translation_batch import TranslationBatchExecutor

from .pipeline_fixtures import batch_plan, fake_pipeline_config


def test_polish_with_length_change_preserves_inline_markers():
    """Polish rewrites Chinese text with significant length changes, inline ranges updated."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    call_count = 0

    def handler(messages, tier, json_mode):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # Translation
            return json.dumps(
                {
                    "translations": [
                        "这是一个[[I_START]]示例[[I_END]]句子。",
                        "另一个[[STRONG_START]]测试[[STRONG_END]]段落。",
                    ]
                }
            )
        elif call_count == 2:  # Polish - simulate length change
            return json.dumps(
                {
                    "polished": [
                        "这是一个完整[[I_START]]示例性[[I_END]]的句子描述内容。",  # Longer
                        "测试的[[STRONG_START]]段落[[STRONG_END]]内容。",  # Shorter, reordered
                    ]
                }
            )
        return "[]"

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    plan = batch_plan(["This is an example sentence.", "Another test paragraph."])

    # Execute with polish=True
    result = executor.execute(plan, polish=True)

    assert len(result.targets) == 2
    assert len(result.inline_metas) == 2

    # First target: "这是一个完整示例性的句子描述内容。" (24 chars)
    # "示例性" should have <i> at correct new positions
    target0 = result.targets[0]
    meta0 = result.inline_metas[0]

    assert "示例性" in target0
    assert "[[I_START]]" not in target0  # Markers removed
    assert "[[I_END]]" not in target0

    assert meta0["version"] == 1
    assert meta0["target_length"] == len(target0)
    ranges0 = meta0["ranges"]
    assert len(ranges0) == 1
    assert ranges0[0]["tag"] == "i"

    # Extract emphasized text using ranges
    r = ranges0[0]
    emphasized_text = target0[r["start"] : r["end"]]
    assert emphasized_text == "示例性"

    # Second target: "测试的段落内容。" (shorter)
    target1 = result.targets[1]
    meta1 = result.inline_metas[1]

    assert "段落" in target1
    assert "[[STRONG_START]]" not in target1
    assert "[[STRONG_END]]" not in target1

    assert meta1["version"] == 1
    ranges1 = meta1["ranges"]
    assert len(ranges1) == 1
    assert ranges1[0]["tag"] == "strong"

    r = ranges1[0]
    emphasized_text = target1[r["start"] : r["end"]]
    assert emphasized_text == "段落"


def test_polish_without_inline_passes_through():
    """Polish without inline markers works normally."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    call_count = 0

    def handler(messages, tier, json_mode):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({"translations": ["普通句子。", "另一个句子。"]})
        elif call_count == 2:
            return json.dumps({"polished": ["精炼的句子。", "优化的句子。"]})
        return "[]"

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    plan = batch_plan(["Normal sentence.", "Another sentence."])

    result = executor.execute(plan, polish=True)

    assert len(result.targets) == 2
    assert result.targets[0] == "精炼的句子。"
    assert result.targets[1] == "优化的句子。"

    # No inline metas or empty
    assert all(not meta or not meta.get("ranges") for meta in result.inline_metas)


def test_polish_with_corrupted_markers_preserves_clean_text():
    """When polish returns corrupted markers, decode warnings raised but text cleaned."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    call_count = 0

    def handler(messages, tier, json_mode):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({"translations": ["这是[[I_START]]示例[[I_END]]。"]})
        elif call_count == 2:  # Polish returns with missing END marker
            return json.dumps({"polished": ["这是[[I_START]]示例的句子。"]})
        return "[]"

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    plan = batch_plan(["This is an example."])
    result = executor.execute(plan, polish=True)

    # Should still produce clean text (markers removed)
    assert "[[I_START]]" not in result.targets[0]
    # Inline meta should be empty or warning recorded (decode will clear corrupted ranges)
    assert not result.inline_metas[0].get("ranges")


def test_polish_continue_path_preserves_inline():
    """Polish continuation (reusing translation turn) also preserves inline."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    call_count = 0

    def handler(messages, tier, json_mode):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({"translations": ["翻译[[STRONG_START]]内容[[STRONG_END]]。"]})
        elif call_count == 2:  # Polish_continue
            return json.dumps({"polished": ["润色后的[[STRONG_START]]文本[[STRONG_END]]内容。"]})
        return "[]"

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    plan = batch_plan(["Translation content."])
    result = executor.execute(plan, polish=True)

    assert len(result.targets) == 1
    assert "[[STRONG_START]]" not in result.targets[0]

    meta = result.inline_metas[0]
    assert meta["version"] == 1
    ranges = meta["ranges"]
    assert len(ranges) == 1
    assert ranges[0]["tag"] == "strong"

    # Extract emphasized part
    r = ranges[0]
    emphasized = result.targets[0][r["start"] : r["end"]]
    assert emphasized == "文本"


def test_polish_with_multiple_inline_tags():
    """Multiple inline tags (i, strong, b) in one translation."""
    config = fake_pipeline_config(state_dir="/tmp/test")

    call_count = 0

    def handler(messages, tier, json_mode):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps(
                {
                    "translations": [
                        "开始[[I_START]]斜体[[I_END]]中间[[STRONG_START]]粗体[[STRONG_END]]结束。"
                    ]
                }
            )
        elif call_count == 2:
            return json.dumps(
                {
                    "polished": [
                        "润色开始[[I_START]]斜体内容[[I_END]]与[[STRONG_START]]加粗部分[[STRONG_END]]的结束。"
                    ]
                }
            )
        return "[]"

    client = FakeClient(handler=handler, config=config.llm)
    translator = Translator(client, config)
    polisher = Polisher(client, config)
    executor = TranslationBatchExecutor(translator, polisher)

    plan = batch_plan(["Start italic middle bold end."])
    result = executor.execute(plan, polish=True)

    target = result.targets[0]
    meta = result.inline_metas[0]

    ranges = meta["ranges"]
    assert len(ranges) == 2

    tags = {r["tag"] for r in ranges}
    assert tags == {"i", "strong"}

    # Verify extracted text
    for r in ranges:
        emphasized = target[r["start"] : r["end"]]
        if r["tag"] == "i":
            assert "斜体内容" == emphasized
        elif r["tag"] == "strong":
            assert "加粗部分" == emphasized
